import os
import json
import time
import threading
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from src.parser import QQChatParser
from src.analyzer import ChatAnalyzer
from src.renderer import HTMLRenderer as ReportRenderer
from src.llm_client import LLMClient
from src.generator import ReportGenerator
from src.history import HistoryManager
from src.registry import (
    DEFAULT_LLM_MAX_OUTPUT_TOKENS,
    DEFAULT_LLM_MAX_RETRIES,
    DEFAULT_LLM_TIMEOUT_SECONDS,
    LLM_MODE_DEFAULT,
    LLM_MODE_CUSTOM,
    COL_DATETIME,
)

# --- Config ---
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
HISTORY_FILE = 'history.json'
CONFIG_FILE = 'config.json'
LOCAL_HISTORY_FILE = 'history.local.json'
LOCAL_CONFIG_FILE = 'config.local.json'
ALLOWED_EXTENSIONS = {'json'}
TOKEN_TO_CHAR_RATIO = 1.5
MAX_INPUT_TOKEN_BUDGET = 1_200_000
SAFE_API_SEGMENT_TOKEN_BUDGET = 900_000
DEFAULT_MAP_CONCURRENCY = 2
MAX_MAP_CONCURRENCY = 16
PENDING_UPLOAD_TTL_SECONDS = 3600
INTERMEDIATE_FOLDER = os.path.join(OUTPUT_FOLDER, 'intermediate')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Flask App ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB limit

# --- Global State for Tasks ---
# In a production app, use Redis/Celery. Here we use a simple dict for local usage.
tasks = {}
pending_uploads = {}
pending_uploads_lock = threading.Lock()

history_manager = HistoryManager(LOCAL_HISTORY_FILE)


def get_active_config_file():
    """Use local settings when present, while keeping the tracked template public-safe."""
    return LOCAL_CONFIG_FILE if os.path.exists(LOCAL_CONFIG_FILE) else CONFIG_FILE

# --- Helpers ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def normalize_years(values):
    """Normalize user-selected years into sorted unique integers."""
    if values is None:
        return []
    if not isinstance(values, (list, tuple, set)):
        values = [values]

    years = []
    for value in values:
        if isinstance(value, bool):
            continue
        try:
            year = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= year <= 9999:
            years.append(year)
    return sorted(set(years))


VALID_REPORT_MODES = {'per_year', 'combined', 'both'}


def normalize_report_mode(value):
    """Normalize the report output mode selected by the user."""
    mode = str(value or 'per_year').strip().lower()
    return mode if mode in VALID_REPORT_MODES else 'per_year'


def build_year_coverage(df, year=None):
    """Describe the actual calendar coverage represented by a DataFrame."""
    if df is None or df.empty:
        return {
            'year': int(year) if year is not None else None,
            'start_time': None,
            'end_time': None,
            'covered_months': [],
            'covered_quarters': [],
            'missing_months': list(range(1, 13)),
            'is_full_year': False,
            'report_scope': 'periodic',
        }

    working = df
    if year is not None:
        working = df[df[COL_DATETIME].dt.year == int(year)]
    if working.empty:
        return build_year_coverage(pd.DataFrame(), year)

    timestamps = working[COL_DATETIME].dropna()
    covered_months = sorted({int(month) for month in timestamps.dt.month})
    covered_quarters = sorted({int(quarter) for quarter in timestamps.dt.quarter})
    missing_months = [month for month in range(1, 13) if month not in covered_months]
    is_full_year = covered_months == list(range(1, 13))
    return {
        'year': int(year) if year is not None else int(timestamps.dt.year.mode().iloc[0]),
        'start_time': timestamps.min().isoformat(),
        'end_time': timestamps.max().isoformat(),
        'covered_months': covered_months,
        'covered_quarters': covered_quarters,
        'missing_months': missing_months,
        'is_full_year': is_full_year,
        'report_scope': 'annual' if is_full_year else 'periodic',
    }


def get_year_summary(df):
    """Return JSON-serializable year ranges and message counts."""
    if df.empty:
        return []

    year_series = df[COL_DATETIME].dt.year
    years = sorted({int(year) for year in year_series.dropna().unique()}, reverse=True)
    summary = []
    for year in years:
        year_df = df[year_series == year]
        coverage = build_year_coverage(year_df, year)
        summary.append({
            'year': year,
            'message_count': int(len(year_df)),
            'start_time': coverage['start_time'],
            'end_time': coverage['end_time'],
            'covered_months': coverage['covered_months'],
            'covered_quarters': coverage['covered_quarters'],
            'missing_months': coverage['missing_months'],
            'is_full_year': coverage['is_full_year'],
        })
    return summary


def cleanup_pending_uploads():
    """Remove inspection files that were uploaded but never started."""
    cutoff = time.time() - PENDING_UPLOAD_TTL_SECONDS
    expired_paths = []
    with pending_uploads_lock:
        for inspection_id, pending in list(pending_uploads.items()):
            if pending['created_at'] < cutoff:
                expired_paths.append(pending['file_path'])
                del pending_uploads[inspection_id]

    for file_path in expired_paths:
        if os.path.exists(file_path):
            os.remove(file_path)


def make_json_safe(value):
    """Convert pandas/numpy values and dates into JSON-serializable values."""
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if hasattr(value, 'item'):
        try:
            return make_json_safe(value.item())
        except (ValueError, TypeError):
            pass
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    return str(value)


def build_year_context(df, meta, year):
    """Build all non-AI report inputs for one calendar year."""
    year = int(year)
    year_df = df[df[COL_DATETIME].dt.year == year].copy()
    if year_df.empty:
        return None

    analyzer = ChatAnalyzer(year_df)
    stats = analyzer.get_basic_stats()
    stats.update(meta)
    stats['total_messages'] = len(year_df)
    stats['selected_years'] = [year]
    stats['start_time'] = year_df[COL_DATETIME].min().isoformat()
    stats['end_time'] = year_df[COL_DATETIME].max().isoformat()
    coverage = build_year_coverage(year_df, year)
    stats['coverage'] = coverage
    stats['report_scope'] = coverage['report_scope']

    return {
        'year': year,
        'df': year_df,
        'analyzer': analyzer,
        'stats': stats,
        'daily_activity': analyzer.get_daily_activity(),
        'rankings': analyzer.get_user_rankings(),
        'hardcore': analyzer.get_hardcore_stats(),
        'coverage': coverage,
    }


def build_combined_context(df, meta, years):
    """Build report inputs for a collection report spanning selected years."""
    if df.empty:
        return None

    analyzer = ChatAnalyzer(df.copy())
    stats = analyzer.get_basic_stats()
    stats.update(meta)
    stats['total_messages'] = len(df)
    stats['selected_years'] = [int(year) for year in years]
    stats['start_time'] = df[COL_DATETIME].min().isoformat()
    stats['end_time'] = df[COL_DATETIME].max().isoformat()
    coverage_by_year = {
        str(year): build_year_coverage(df, year)
        for year in years
    }
    coverage = {
        'report_scope': 'combined',
        'years': [int(year) for year in years],
        'coverage_by_year': coverage_by_year,
        'is_full_year': all(
            item['is_full_year'] for item in coverage_by_year.values()
        ),
    }
    stats['coverage'] = coverage
    stats['report_scope'] = 'combined'

    return {
        'years': [int(year) for year in years],
        'df': df.copy(),
        'analyzer': analyzer,
        'stats': stats,
        'daily_activity': analyzer.get_daily_activity(),
        'rankings': analyzer.get_user_rankings(),
        'hardcore': analyzer.get_hardcore_stats(),
        'coverage': coverage,
    }


def serialize_year_context(context):
    """Persist the non-AI inputs needed to render a yearly report later."""
    return {
        'stats': make_json_safe(context['stats']),
        'daily_activity': make_json_safe(
            context['daily_activity'].to_dict('records')
        ),
        'rankings': {
            name: make_json_safe(frame.to_dict('records'))
            for name, frame in context['rankings'].items()
        },
        'hardcore': make_json_safe(context['hardcore']),
        'coverage': make_json_safe(context.get('coverage', {})),
    }


def serialize_map_job(job):
    """Keep only JSON-safe identifying fields in an intermediate artifact."""
    return {
        'order': int(job.get('order', -1)),
        'report_year': job.get('report_year'),
        'name': job.get('name', ''),
        'is_segmented': bool(job.get('is_segmented', False)),
    }


def save_intermediate_artifact(
    task_id,
    report_years,
    map_results_by_year,
    year_contexts,
    status,
    total_map_jobs,
    completed_map_jobs,
    report_mode='per_year',
    failures=None,
    pending_jobs=None,
    successful_map_jobs=None,
):
    """Atomically save Map diagnostics for the current task."""
    os.makedirs(INTERMEDIATE_FOLDER, exist_ok=True)
    artifact_path = os.path.join(INTERMEDIATE_FOLDER, f'{task_id}.json')
    failures = list(failures or [])
    pending_jobs = [serialize_map_job(job) for job in (pending_jobs or [])]
    successful_map_jobs = (
        int(successful_map_jobs)
        if successful_map_jobs is not None
        else max(0, int(completed_map_jobs) - len(failures))
    )
    artifact = {
        'version': 2,
        'task_id': task_id,
        'status': status,
        'report_mode': normalize_report_mode(report_mode),
        'report_years': [int(year) for year in report_years],
        'completed_map_jobs': int(completed_map_jobs),
        'successful_map_jobs': successful_map_jobs,
        'failed_map_job_count': len(failures),
        'total_map_jobs': int(total_map_jobs),
        'map_results': make_json_safe(map_results_by_year),
        'failures': make_json_safe(failures),
        'pending_jobs': make_json_safe(pending_jobs),
        'pending_map_jobs': make_json_safe(pending_jobs),
        'failed_map_jobs': make_json_safe(failures),
        'year_contexts': {
            str(year): serialize_year_context(context)
            for year, context in year_contexts.items()
        },
    }
    temp_path = f'{artifact_path}.tmp'
    with open(temp_path, 'w', encoding='utf-8') as handle:
        json.dump(artifact, handle, ensure_ascii=False, indent=2)
    os.replace(temp_path, artifact_path)
    return artifact_path


def load_intermediate_artifact(artifact_path):
    """Read and validate a persisted Map diagnostic artifact before Reduce."""
    with open(artifact_path, 'r', encoding='utf-8') as handle:
        artifact = json.load(handle)
    if artifact.get('version') not in {1, 2}:
        raise ValueError('不支持的中间产物版本')
    if not isinstance(artifact.get('map_results'), dict):
        raise ValueError('中间产物缺少 Map 结果')
    artifact.setdefault('failures', [])
    artifact.setdefault('pending_jobs', [])
    artifact.setdefault('pending_map_jobs', artifact['pending_jobs'])
    artifact.setdefault('failed_map_jobs', artifact['failures'])
    return artifact


def build_reduce_stats(context, years, map_failures=None):
    """Build Reduce input from one year or a selected year collection."""
    if isinstance(years, (list, tuple, set)):
        selected_years = [int(year) for year in years]
    else:
        selected_years = [int(years)]

    stats = context['stats']
    hardcore = context.get('hardcore') or {}
    return {
        'total_messages': stats.get('total_messages', 0),
        'total_users': stats.get('total_users', 0),
        'year': '、'.join(map(str, selected_years)),
        'selected_years': selected_years,
        'active_users_count': stats.get(
            'active_users_count', stats.get('total_users', 0)
        ),
        'silent_users_count': stats.get('silent_users_count', 0),
        'top_talkers': hardcore.get('top_talkers', []),
        'top_repeaters': stats.get('top_repeaters', []),
        'hardcore': hardcore,
        'coverage': context.get('coverage', {}),
        'report_scope': context.get('stats', {}).get('report_scope'),
        'map_failures': make_json_safe(map_failures or []),
        'map_data_complete': not bool(map_failures),
    }

class TaskLogger:
    def __init__(self, task_id):
        self.task_id = task_id
        self._lock = threading.Lock()
    
    def info(self, msg):
        with self._lock:
            if self.task_id in tasks:
                tasks[self.task_id]['logs'].append(msg)
                print(f"[Task {self.task_id}] {msg}")

    def progress(self, percent, status_text):
        with self._lock:
            if self.task_id in tasks:
                tasks[self.task_id]['progress'] = percent
                tasks[self.task_id]['status_text'] = status_text

def smart_sample(df, max_tokens, logger=None):
    """
    智能采样函数，确保不超过 Token 预算。
    """
    # 估算字符限制 (1 Token ≈ 1.5 Chars)
    target_chars = max(1, int(max_tokens * TOKEN_TO_CHAR_RATIO))

    def select_evenly(items, count):
        """Select a chronological spread while keeping the first and last messages."""
        if count >= len(items):
            return items
        if count <= 1:
            return [items[0]]

        last_index = len(items) - 1
        return [items[(index * last_index) // (count - 1)] for index in range(count)]

    def join_messages(items):
        return "\n".join(items)
    
    # 预处理消息格式
    if 'formatted_msg' not in df.columns:
        df['formatted_msg'] = df.apply(
            lambda x: f"[{str(x['datetime'])[:16]}] {x.get('user_name', 'Unknown')}: {str(x['content'])[:100]}", 
            axis=1
        )
    
    full_text_list = df['formatted_msg'].tolist()
    total_msgs = len(full_text_list)
    
    if total_msgs == 0:
        return ""

    estimated_total_chars = sum(len(m) for m in full_text_list) + max(0, total_msgs - 1)

    if estimated_total_chars <= target_chars:
        sampled_msgs = full_text_list
        sample_text = join_messages(sampled_msgs)
        if logger: logger.info(f"数据量较小 ({len(sample_text)} chars)，全量发送")
    else:
        # Estimate a message count from the actual formatted text, then adjust
        # until the joined result really fits the budget. The old floor-based
        # stride could become 1 and accidentally send the complete oversized set.
        target_msg_count = max(1, min(
            total_msgs,
            int(total_msgs * target_chars / estimated_total_chars),
        ))
        sampled_msgs = select_evenly(full_text_list, target_msg_count)
        sample_text = join_messages(sampled_msgs)

        while len(sample_text) > target_chars and target_msg_count > 1:
            next_count = max(1, int(target_msg_count * target_chars / len(sample_text)))
            if next_count >= target_msg_count:
                next_count = target_msg_count - 1
            target_msg_count = next_count
            sampled_msgs = select_evenly(full_text_list, target_msg_count)
            sample_text = join_messages(sampled_msgs)

        # A single unusually long message is still bounded by the budget.
        if len(sample_text) > target_chars:
            sample_text = sample_text[:target_chars]

        if logger:
            logger.info(
                f"数据量过大，执行均匀采样。从 {total_msgs} 条中抽取 {len(sampled_msgs)} 条，"
                f"实际 {len(sample_text)} chars（目标不超过 {target_chars} chars）。"
            )

    return sample_text


def split_text_by_token_budget(text, max_tokens):
    """Split formatted chat text at message boundaries for one API request."""
    if not text:
        return [""]

    max_chars = max(1, int(max_tokens * TOKEN_TO_CHAR_RATIO))
    if len(text) <= max_chars:
        return [text]

    segments = []
    current_lines = []
    current_chars = 0

    def flush_current():
        if current_lines:
            segments.append("\n".join(current_lines))

    for line in text.splitlines():
        # Formatted messages are short, but keep a hard fallback for unusual input.
        if len(line) > max_chars:
            flush_current()
            current_lines.clear()
            current_chars = 0
            for start in range(0, len(line), max_chars):
                segments.append(line[start:start + max_chars])
            continue

        added_chars = len(line) + (1 if current_lines else 0)
        if current_lines and current_chars + added_chars > max_chars:
            flush_current()
            current_lines.clear()
            current_chars = 0

        current_lines.append(line)
        current_chars += len(line) + (1 if len(current_lines) > 1 else 0)

    flush_current()
    return segments or [text[:max_chars]]

# --- Analysis Worker ---
def run_analysis_task(task_id, file_path, config):
    logger = TaskLogger(task_id)
    result_urls = []
    failed_reports = []
    try:
        tasks[task_id]['state'] = 'processing'
        logger.progress(5, "正在初始化组件...")
        
        # 1. Parse
        logger.info(f"正在解析文件: {os.path.basename(file_path)}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            parser = QQChatParser()
            df, meta = parser.parse_json(content)
        except Exception as e:
            raise ValueError(f"文件解析失败: {str(e)}")
        
        logger.progress(20, f"解析完成，共加载 {len(df)} 条消息")
        logger.info(f"解析成功: {len(df)} messages")

        selected_years = normalize_years(config.get('selected_years'))
        report_mode = normalize_report_mode(config.get('report_mode'))
        if selected_years:
            available_years = {
                int(year) for year in df[COL_DATETIME].dt.year.dropna().unique()
            }
            missing_years = [year for year in selected_years if year not in available_years]
            if missing_years:
                raise ValueError(
                    f"所选年份不存在于文件中: {', '.join(map(str, missing_years))}"
                )

            df = df[df[COL_DATETIME].dt.year.isin(selected_years)].copy()
            if df.empty:
                raise ValueError("所选年份没有可分析的有效消息")

            logger.progress(25, f"已选择 {len(selected_years)} 个年份，共 {len(df)} 条消息")
            logger.info(
                f"AI分析年份: {', '.join(map(str, selected_years))}；"
                f"筛选后 {len(df)} messages"
            )

        # 2. Analyze (Stats), keeping one independent report context per year.
        logger.progress(30, "正在进行统计分析...")
        analyzer = ChatAnalyzer(df)
        report_years = selected_years or [int(analyzer.get_target_year())]
        if not selected_years:
            # The legacy upload path analyzes the dominant year only. Filter before
            # splitting so dynamic sampling cannot mix neighboring calendar years.
            target_year = report_years[0]
            df = df[df[COL_DATETIME].dt.year == target_year].copy()
            analyzer = ChatAnalyzer(df)
        year_contexts = {}
        for report_year in report_years:
            context = build_year_context(df, meta, report_year)
            if context:
                year_contexts[int(report_year)] = context

        if not year_contexts:
            raise ValueError("没有可生成报告的有效年份")

        report_years = [
            int(year) for year in report_years if int(year) in year_contexts
        ]
        combined_df = df[df[COL_DATETIME].dt.year.isin(report_years)].copy()
        combined_context = build_combined_context(
            combined_df,
            meta,
            report_years,
        )
        if not combined_context:
            raise ValueError("没有可生成集合报告的有效数据")
        logger.progress(40, "统计分析完成")
        logger.info(
            f"已建立 {len(report_years)} 个年度报告上下文："
            f"{', '.join(map(str, report_years))}；报告模式: {report_mode}"
        )

        # 3. AI Analysis (Map-Reduce)
        logger.progress(45, "正在初始化 AI 分析组件...")
        
        request_timeout = int(config.get('request_timeout_seconds', DEFAULT_LLM_TIMEOUT_SECONDS))
        request_output_tokens = int(config.get('request_max_output_tokens', DEFAULT_LLM_MAX_OUTPUT_TOKENS))
        request_retries = int(config.get('request_max_retries', DEFAULT_LLM_MAX_RETRIES))
        llm_mode = config.get('mode', LLM_MODE_DEFAULT)

        llm_config = {
            'mode': llm_mode,
            'timeout': request_timeout,
            'max_output_tokens': request_output_tokens,
            'max_retries': request_retries,
            'token_parameter': config.get('token_parameter', 'auto'),
            'logger': logger.info,
        }
        if llm_mode == LLM_MODE_CUSTOM:
            llm_config.update({
                'api_key': config.get('api_key'),
                'base_url': config.get('base_url'),
                'model': config.get('model'),
            })
        else:
            logger.info("使用内置演示模式 (Mock)")

        logger.info(
            f"LLM请求配置: mode={llm_mode}, timeout={request_timeout}s, "
            f"output_limit={request_output_tokens} tokens, retries={request_retries}"
        )
            
        # Get specific models for each phase
        model_map = config.get('model_map') or llm_config.get('model')
        model_reduce = config.get('model_reduce') or llm_config.get('model')
        model_refine = config.get('model_refine') or llm_config.get('model')

        client = LLMClient(**llm_config)
        generator = ReportGenerator(client, logger=logger.info)
        configured_max_tokens = int(config.get('max_tokens', 128000))
        max_tokens = max(128000, min(configured_max_tokens, MAX_INPUT_TOKEN_BUDGET))
        if configured_max_tokens != max_tokens:
            logger.info(
                f"输入预算已限制为 {max_tokens} tokens（可选范围 128k - 1200k）"
            )
        logger.info(
            f"本地输入采样预算: {max_tokens} tokens（不等于 API 输出上限；"
            f"单次 API 请求最多约 {SAFE_API_SEGMENT_TOKEN_BUDGET} tokens）"
        )

        configured_concurrency = int(config.get('max_concurrency', DEFAULT_MAP_CONCURRENCY))
        map_concurrency = max(1, min(configured_concurrency, MAX_MAP_CONCURRENCY))
        if configured_concurrency != map_concurrency:
            logger.info(
                f"Map 并发数已限制为 {map_concurrency}（可选范围 1 - {MAX_MAP_CONCURRENCY}）"
            )
        logger.info(f"Map 并发请求数: {map_concurrency}")

        # Step 1: Map (quarterly analysis grouped by report year)
        logger.info("正在进行切分...")
        splits = analyzer.get_quarterly_splits(selected_years or None)

        if not splits:
            logger.info("切分失败，按年度降级为全量分析")
            splits = {
                f"{year}_Whole_Year": context['df']
                for year, context in year_contexts.items()
            }

        map_jobs = []
        total_quarters = len(splits)
        processed_count = 0
        for q_name, q_df in splits.items():
            processed_count += 1
            progress_start = 50 + int(
                (processed_count - 1) / max(1, total_quarters) * 30
            )
            logger.progress(
                progress_start,
                f"正在准备 {q_name} ({processed_count}/{total_quarters})..."
            )

            if q_df.empty:
                logger.info(f"分块 {q_name} 数据为空，跳过")
                continue

            split_years = q_df[COL_DATETIME].dt.year.dropna()
            report_year = int(split_years.iloc[0]) if not split_years.empty else None
            if report_year not in year_contexts:
                logger.info(f"无法确定 {q_name} 所属年份，跳过")
                continue

            coverage = year_contexts[report_year].get('coverage', {})

            # Sample using Adaptive Strategy (Phase 2 - 3.3)
            sample_text = smart_sample(q_df, max_tokens, logger)

            # Keep each request below the tested provider capacity. A larger
            # selected budget is handled as multiple chronological Map segments.
            segment_budget = min(max_tokens, SAFE_API_SEGMENT_TOKEN_BUDGET)
            segments = split_text_by_token_budget(sample_text, segment_budget)
            if len(segments) > 1:
                logger.info(
                    f"{q_name} 内容超过单次 API 容量，分为 {len(segments)} 段发送 "
                    f"（每段不超过约 {segment_budget} tokens）"
                )

            for segment_index, segment_text in enumerate(segments, start=1):
                segment_name = q_name
                if len(segments) > 1:
                    segment_name = f"{q_name}（第 {segment_index}/{len(segments)} 段）"

                map_jobs.append({
                    'order': len(map_jobs),
                    'report_year': report_year,
                    'name': segment_name,
                    'text': segment_text,
                    'is_segmented': len(segments) > 1,
                    'coverage': coverage,
                    'is_periodic': not coverage.get('is_full_year', False),
                })

        if not map_jobs:
            raise ValueError("没有可发送的年度聊天分片")

        map_results_by_year = {str(year): [] for year in report_years}
        intermediate_path = save_intermediate_artifact(
            task_id,
            report_years,
            map_results_by_year,
            year_contexts,
            status='map_running',
            total_map_jobs=len(map_jobs),
            completed_map_jobs=0,
            report_mode=report_mode,
            pending_jobs=map_jobs,
        )
        tasks[task_id]['intermediate_path'] = intermediate_path
        logger.info(f"Map 中间产物已创建: {intermediate_path}")

        def run_map_job(job):
            segment_name = job['name']
            logger.info(f"发送 AI 请求: {segment_name} (Model: {model_map})")
            result = generator.generate_quarterly_analysis(
                segment_name,
                job['text'],
                model=model_map,
                is_periodic=job['is_periodic'],
                coverage=job['coverage'],
            )
            if job['is_segmented'] and isinstance(result, dict):
                result = dict(result)
                result['_source_segment'] = segment_name
            return job['order'], result

        if map_jobs:
            logger.info(f"开始并行执行 {len(map_jobs)} 个 Map 请求")
            with ThreadPoolExecutor(max_workers=map_concurrency) as executor:
                future_to_job = {
                    executor.submit(run_map_job, job): job
                    for job in map_jobs
                }
                pending_orders = {job['order'] for job in map_jobs}
                failures = []
                successful_jobs = 0
                completed_jobs = 0
                for future in as_completed(future_to_job):
                    job = future_to_job[future]
                    try:
                        job_order, result = future.result()
                        year_key = str(job['report_year'])
                        map_results_by_year[year_key].append({
                            'order': job_order,
                            'period': job['name'],
                            'analysis': result,
                        })
                        successful_jobs += 1
                        logger.info(f"Map 完成: {job['name']}")
                    except Exception as exc:
                        failure = {
                            'order': job['order'],
                            'report_year': job['report_year'],
                            'name': job['name'],
                            'error_type': type(exc).__name__,
                            'error': str(exc),
                            'attempts': request_retries,
                        }
                        failures.append(failure)
                        logger.info(
                            f"Map 失败但继续收集其他结果: {job['name']} | "
                            f"{type(exc).__name__}: {exc}"
                        )
                    finally:
                        pending_orders.discard(job['order'])
                        completed_jobs += 1
                    pending_jobs = [
                        item for item in map_jobs
                        if item['order'] in pending_orders
                    ]
                    if pending_jobs:
                        map_status = 'map_partial' if failures else 'map_running'
                    elif successful_jobs == 0:
                        map_status = 'map_failed'
                    elif failures:
                        map_status = 'map_partial'
                    else:
                        map_status = 'map_complete'
                    progress = 50 + int(completed_jobs / len(map_jobs) * 30)
                    logger.progress(
                        progress,
                        f"Map 分析完成 ({completed_jobs}/{len(map_jobs)})..."
                    )
                    save_intermediate_artifact(
                        task_id,
                        report_years,
                        map_results_by_year,
                        year_contexts,
                        status=map_status,
                        total_map_jobs=len(map_jobs),
                        completed_map_jobs=completed_jobs,
                        report_mode=report_mode,
                        failures=failures,
                        pending_jobs=pending_jobs,
                        successful_map_jobs=successful_jobs,
                    )

        for results in map_results_by_year.values():
            results.sort(key=lambda item: item['order'])
        if 'failures' not in locals():
            failures = []
        if 'successful_jobs' not in locals():
            successful_jobs = 0
        final_map_status = (
            'map_failed'
            if successful_jobs == 0
            else 'map_partial'
            if failures
            else 'map_complete'
        )
        save_intermediate_artifact(
            task_id,
            report_years,
            map_results_by_year,
            year_contexts,
            status=final_map_status,
            total_map_jobs=len(map_jobs),
            completed_map_jobs=len(map_jobs),
            report_mode=report_mode,
            failures=failures,
            pending_jobs=[],
            successful_map_jobs=successful_jobs,
        )

        # Make Reduce consume the persisted diagnostic artifact from this task.
        intermediate = load_intermediate_artifact(intermediate_path)
        map_results_by_year = intermediate['map_results']
        map_failures = intermediate.get('failures', [])
        logger.info(
            f"已加载 Map 中间产物：{intermediate['completed_map_jobs']}/"
            f"{intermediate['total_map_jobs']} 个请求"
        )

        # Step 2-5: Build the requested output set. Independent yearly reports
        # and the legacy collection report share the same persisted Map output.
        anime_theme = config.get('anime_theme', 'default')
        custom_theme_prompt = config.get('custom_theme_prompt', '')
        renderer = ReportRenderer()
        def analyses_for_year(report_year):
            entries = map_results_by_year.get(str(report_year), [])
            return [
                entry.get('analysis', entry)
                if isinstance(entry, dict) else entry
                for entry in entries
            ]

        report_specs = []
        if report_mode in {'per_year', 'both'}:
            for report_year in report_years:
                year_context = year_contexts[report_year]
                is_year_periodic = not year_context.get('coverage', {}).get(
                    'is_full_year', False
                )
                report_specs.append({
                    'kind': 'year',
                    'year': int(report_year),
                    'label': f'{report_year} 年度报告',
                    'context': year_context,
                    'map_results': analyses_for_year(report_year),
                    'reduce_years': [int(report_year)],
                    'is_periodic': is_year_periodic,
                    'report_scope': 'periodic' if is_year_periodic else 'annual',
                    'filename_suffix': str(report_year),
                })

        if report_mode in {'combined', 'both'}:
            combined_results = []
            for report_year in report_years:
                combined_results.extend(analyses_for_year(report_year))
            year_label = '、'.join(map(str, report_years))
            collection_label = (
                f'{year_label} 多年集合报告'
                if len(report_years) > 1
                else f'{year_label} 集合报告'
            )
            report_specs.append({
                'kind': 'combined',
                'year': None,
                'label': collection_label,
                'context': combined_context,
                'map_results': combined_results,
                'reduce_years': report_years,
                'is_periodic': (
                    len(report_years) > 1
                    or not combined_context.get('coverage', {}).get(
                        'is_full_year', False
                    )
                ),
                'report_scope': 'combined',
                'filename_suffix': 'combined',
            })

        if not report_specs:
            raise ValueError('没有选择有效的报告生成模式')

        total_reports = len(report_specs)

        def failures_for_spec(report_spec):
            if report_spec['kind'] == 'combined':
                return list(map_failures)
            return [
                failure for failure in map_failures
                if int(failure.get('report_year', -1)) == int(report_spec['year'])
            ]

        for report_index, report_spec in enumerate(report_specs, start=1):
            report_label = report_spec['label']
            context = report_spec['context']
            report_results = report_spec['map_results']
            report_failures = failures_for_spec(report_spec)
            if not report_results:
                failure = {
                    'phase': 'reduce',
                    'kind': report_spec['kind'],
                    'year': report_spec['year'],
                    'label': report_label,
                    'error_type': 'NoMapResults',
                    'error': f"{report_label} 没有可用于汇总的 Map 结果",
                }
                failed_reports.append(failure)
                logger.info(failure['error'])
                continue

            logger.progress(
                85 + int((report_index - 1) / total_reports * 10),
                f"正在生成 {report_label} ({report_index}/{total_reports})..."
            )
            try:
                logger.info(
                    f"发送 AI 请求: {report_label} (Model: {model_reduce})"
                )
                reduce_stats = build_reduce_stats(
                    context,
                    report_spec['reduce_years'],
                    map_failures=report_failures,
                )
                final_html = generator.generate_annual_report(
                    report_results,
                    reduce_stats,
                    anime_theme=anime_theme,
                    custom_theme_prompt=custom_theme_prompt,
                    model=model_reduce,
                    is_periodic=report_spec['is_periodic'],
                    report_scope=report_spec['report_scope'],
                )
                logger.info(f"{report_label} 汇总完成")

                logger.progress(
                    90 + int(report_index / total_reports * 5),
                    f"正在渲染 {report_label} HTML..."
                )
                chat_name = context['stats'].get('chat_name', 'QQ聊天记录')
                report_filename = f"report_{task_id}_{report_spec['filename_suffix']}.html"
                report_path = os.path.join(OUTPUT_FOLDER, report_filename)
                renderer.render(
                    stats=context['stats'],
                    daily_activity=context['daily_activity'],
                    summary=final_html,
                    rankings=context['rankings'],
                    output_path=report_path
                )

                if config.get('enhance_mode', False):
                    logger.progress(
                        95 + int(report_index / total_reports * 3),
                        f"正在增强 {report_label} HTML..."
                    )
                    logger.info(
                        f"启动 {report_label} HTML 修复与 CSS 优化 "
                        f"(Model: {model_refine})"
                    )
                    try:
                        with open(report_path, 'r', encoding='utf-8') as handle:
                            raw_html = handle.read()
                        refined_html = generator.refine_report_html(
                            raw_html,
                            model=model_refine
                        )
                        if refined_html and len(refined_html) > 100:
                            with open(report_path, 'w', encoding='utf-8') as handle:
                                handle.write(refined_html)
                            logger.info(f"{report_label} HTML 增强完成并已保存")
                        else:
                            logger.info(f"{report_label} HTML 增强结果异常，保留原文件")
                    except Exception as exc:
                        logger.info(f"{report_label} HTML 增强失败: {exc}")

                history_manager.add_record(
                    chat_name=chat_name,
                    messages_count=context['stats']['total_messages'],
                    report_path=report_path,
                    year=report_spec['year'],
                    report_mode=report_spec['kind'],
                )
                result_urls.append({
                    'year': report_spec['year'],
                    'years': report_spec['reduce_years'],
                    'kind': report_spec['kind'],
                    'label': report_label,
                    'url': f'/download/{report_filename}',
                    'filename': report_filename,
                })
                tasks[task_id]['result_urls'] = list(result_urls)
                tasks[task_id]['result_url'] = result_urls[0]['url']
            except Exception as exc:
                failure = {
                    'phase': 'reduce',
                    'kind': report_spec['kind'],
                    'year': report_spec['year'],
                    'label': report_label,
                    'error_type': type(exc).__name__,
                    'error': str(exc),
                }
                failed_reports.append(failure)
                logger.info(
                    f"{report_label} 失败但继续处理其他报告: "
                    f"{type(exc).__name__}: {exc}"
                )

        all_failures = list(map_failures) + list(failed_reports)
        tasks[task_id]['result_urls'] = result_urls
        tasks[task_id]['result_url'] = result_urls[0]['url'] if result_urls else None
        tasks[task_id]['failed_reports'] = failed_reports
        tasks[task_id]['failures'] = all_failures
        if result_urls and not all_failures:
            tasks[task_id]['state'] = 'completed'
            tasks[task_id]['error'] = None
            logger.progress(100, f"{len(result_urls)} 份报告生成完成！")
        elif result_urls:
            tasks[task_id]['state'] = 'partial'
            tasks[task_id]['error'] = None
            logger.progress(
                100,
                f"已生成 {len(result_urls)} 份报告，另有 {len(all_failures)} 项失败"
            )
        else:
            tasks[task_id]['state'] = 'failed'
            tasks[task_id]['error'] = (
                f"没有报告生成成功；失败项 {len(all_failures)} 个"
            )
            logger.progress(100, tasks[task_id]['error'])

    except Exception as e:
        logger.info(f"Error: {str(e)}")
        outer_failure = {
            'phase': 'task',
            'error_type': type(e).__name__,
            'error': str(e),
        }
        if result_urls:
            tasks[task_id]['state'] = 'partial'
            tasks[task_id]['result_urls'] = list(result_urls)
            tasks[task_id]['result_url'] = result_urls[0]['url']
            tasks[task_id]['failed_reports'] = list(failed_reports)
            tasks[task_id]['failures'] = (
                list(tasks[task_id].get('failures', []))
                + list(failed_reports)
                + [outer_failure]
            )
            tasks[task_id]['error'] = None
        else:
            tasks[task_id]['state'] = 'failed'
            tasks[task_id]['error'] = str(e)
            tasks[task_id]['failures'] = (
                list(tasks[task_id].get('failures', []))
                + list(failed_reports)
                + [outer_failure]
            )
    finally:
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)


def start_analysis_task(file_path, config):
    """Create and launch an analysis task for an already-saved JSON file."""
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        'state': 'queued',
        'progress': 0,
        'status_text': '等待队列...',
        'logs': [],
        'result_url': None,
        'result_urls': [],
        'failed_reports': [],
        'failures': [],
        'report_mode': normalize_report_mode(config.get('report_mode')),
        'intermediate_path': None,
        'error': None
    }

    thread = threading.Thread(
        target=run_analysis_task,
        args=(task_id, file_path, config),
        daemon=True,
    )
    thread.start()
    return task_id


# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/inspect', methods=['POST'])
def inspect_file():
    """Upload and inspect a chat export before starting AI analysis."""
    cleanup_pending_uploads()

    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'status': 'error', 'message': 'Invalid file type'}), 400

    filename = secure_filename(file.filename) or 'chat.json'
    inspection_id = str(uuid.uuid4())
    save_path = os.path.join(
        app.config['UPLOAD_FOLDER'],
        f"inspection_{inspection_id}_{filename}"
    )

    try:
        file.save(save_path)
        with open(save_path, 'r', encoding='utf-8') as handle:
            content = handle.read()

        df, meta = QQChatParser().parse_json(content)
        years = get_year_summary(df)
        if not years:
            raise ValueError('没有识别到带有效时间的聊天记录')

        default_year = max(
            years,
            key=lambda item: (item['message_count'], item['year'])
        )['year']

        with pending_uploads_lock:
            pending_uploads[inspection_id] = {
                'file_path': save_path,
                'filename': filename,
                'years': [item['year'] for item in years],
                'created_at': time.time(),
            }

        return jsonify({
            'status': 'success',
            'inspection_id': inspection_id,
            'chat_name': meta.get('chat_name'),
            'total_messages': len(df),
            'years': years,
            'default_year': default_year,
        })
    except Exception as exc:
        if os.path.exists(save_path):
            os.remove(save_path)
        return jsonify({
            'status': 'error',
            'message': f'文件识别失败: {exc}',
        }), 400


@app.route('/api/analyze', methods=['POST'])
def analyze():
    # New two-step flow: use an inspected upload and an explicit year selection.
    if request.is_json:
        request_data = request.get_json(silent=True) or {}
        inspection_id = request_data.get('inspection_id')
        if inspection_id:
            config = request_data.get('config') or {}
            selected_years = normalize_years(request_data.get('selected_years'))
            if not isinstance(config, dict):
                return jsonify({'status': 'error', 'message': 'Invalid config'}), 400
            if not selected_years:
                return jsonify({'status': 'error', 'message': '请至少选择一个年份'}), 400

            cleanup_pending_uploads()
            with pending_uploads_lock:
                pending = pending_uploads.get(inspection_id)

            if not pending:
                return jsonify({
                    'status': 'error',
                    'message': '文件识别已过期，请重新上传文件',
                }), 400

            available_years = set(pending['years'])
            invalid_years = [
                year for year in selected_years if year not in available_years
            ]
            if invalid_years:
                return jsonify({
                    'status': 'error',
                    'message': f"所选年份无效: {', '.join(map(str, invalid_years))}",
                }), 400

            with pending_uploads_lock:
                pending = pending_uploads.pop(inspection_id, None)
            if not pending:
                return jsonify({
                    'status': 'error',
                    'message': '该文件已经开始分析或已失效，请重新上传',
                }), 400

            config = dict(config)
            config['selected_years'] = selected_years
            try:
                task_id = start_analysis_task(pending['file_path'], config)
            except Exception:
                if os.path.exists(pending['file_path']):
                    os.remove(pending['file_path'])
                raise
            return jsonify({'status': 'success', 'task_id': task_id})

    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'})
    
    file = request.files['file']
    config_str = request.form.get('config', '{}')
    
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'})
    
    if file and allowed_file(file.filename):
        try:
            config = json.loads(config_str)
            filename = secure_filename(file.filename)
            task_id = str(uuid.uuid4())
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{filename}")
            file.save(save_path)
            task_id = start_analysis_task(save_path, config)
            
            return jsonify({'status': 'success', 'task_id': task_id})
            
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)})
            
    return jsonify({'status': 'error', 'message': 'Invalid file type'})

@app.route('/api/status/<task_id>')
def task_status(task_id):
    if task_id not in tasks:
        return jsonify({'status': 'error', 'message': 'Task not found'}), 404
    
    task = tasks[task_id]
    # Return logs and clear them from server memory to avoid duplication if client polls?
    # Actually simple polling: client maintains offset or we just send all?
    # For simplicity, send all logs but client filters? Or we pop?
    # Let's just send last 5 logs or new logs.
    # To keep it stateless for client: client just displays what it gets.
    # We will implement a simple "read and clear" for logs here?
    # No, multiple polls might miss. Let's just send all logs and client appends unique? 
    # Or simpler: send "new_logs" by keeping track of read index? Too complex.
    # Let's just return all logs for now, client clears box and re-renders or appends diff.
    # Optimization: Client clears and appends.
    
    # Better: return "new_logs" by checking an optional "last_log_index" param?
    # Let's keep it extremely simple: Send all logs, client handles it. 
    # Or actually, we pop logs! Because logs are ephemeral stream.
    logs_to_send = list(task['logs']) # Copy
    task['logs'] = [] # Clear sent logs
    
    return jsonify({
        'state': task['state'],
        'progress': task['progress'],
        'status_text': task['status_text'],
        'new_logs': logs_to_send,
        'result_url': task['result_url'],
        'result_urls': task.get('result_urls', []),
        'failed_reports': task.get('failed_reports', []),
        'failures': task.get('failures', []),
        'report_mode': task.get('report_mode'),
        'intermediate_path': task.get('intermediate_path'),
        'error': task['error']
    })

@app.route('/api/history')
def get_history():
    return jsonify(history_manager.get_records())

@app.route('/api/config', methods=['GET', 'POST'])
def handle_config():
    if request.method == 'POST':
        try:
            config_data = request.json
            with open(LOCAL_CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2)
            return jsonify({'status': 'success', 'message': 'Config saved'})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)})
    else:
        try:
            config_path = get_active_config_file()
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    return jsonify(json.load(f))
            return jsonify({})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/test_connection', methods=['POST'])
def test_connection_api():
    """
    测试 LLM API 连接路由
    """
    try:
        config = request.json
        api_key = config.get('api_key')
        base_url = config.get('base_url')
        model = config.get('model')
        mode = config.get('mode', 'default')
        
        print(f"[Debug] Received Test Connection Request: Mode={mode}, BaseURL={base_url}, Model={model}")
        
        # 临时实例化 Client 进行测试
        client = LLMClient(mode=mode, api_key=api_key, base_url=base_url, model=model)
        
        result = client.test_connection()
        return jsonify(result)
        
    except Exception as e:
        print(f"[Error] Test Connection Failed: {e}")
        return jsonify({'success': False, 'message': f"服务器内部错误: {str(e)}"})

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    debug_enabled = os.environ.get('QQ_ANALYZER_DEBUG', '').lower() in {
        '1', 'true', 'yes', 'on'
    }
    print(
        f"Starting Flask Server at http://localhost:5000 "
        f"(debug={debug_enabled}, reloader={debug_enabled})"
    )
    app.run(
        debug=debug_enabled,
        use_reloader=debug_enabled,
        threaded=True,
        port=5000,
    )
