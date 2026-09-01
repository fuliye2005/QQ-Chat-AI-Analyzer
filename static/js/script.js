document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM loaded, initializing...");
    initUI();
    loadHistory();
    loadConfig();
});

let pendingInspectionId = null;

function initUI() {
    console.log("Initializing UI...");
    // 1. Sidebar Toggles
    window.toggleSidebar = (side) => {
        console.log("toggleSidebar", side);
        const el = document.getElementById(`sidebar-${side}`);
        if (el) el.classList.toggle('active');
        else console.error(`Sidebar not found: sidebar-${side}`);
    };

    // 2. Modal Toggles
    window.toggleModal = (id) => {
        console.log("toggleModal", id);
        const el = document.getElementById(`modal-${id}`);
        if (el) el.classList.toggle('active');
        else console.error(`Modal not found: modal-${id}`);
    };
    
    // Close modal on outside click
    document.querySelectorAll('.modal-overlay').forEach(overlay => {
        overlay.addEventListener('click', (e) => {
            // Prevent closing tutorial modal when clicking outside
            if (e.target === overlay && overlay.id !== 'modal-tutorial') {
                overlay.classList.remove('active');
            }
        });
    });

    // 3. Config Toggle
    window.toggleCustomConfig = () => {
        const mode = document.getElementById('llm-mode').value;
        document.getElementById('custom-config-area').style.display = 
            mode === 'custom' ? 'block' : 'none';
    };

    // 3.1 Theme Toggle
    window.toggleCustomTheme = () => {
        const theme = document.getElementById('anime-theme').value;
        document.getElementById('custom-theme-area').style.display = 
            theme === 'custom' ? 'block' : 'none';
    };

    // 4. Token Slider
    const slider = document.getElementById('sampling-strength');
    const output = document.getElementById('token-val');
    if (slider && output) {
        slider.oninput = function() {
            output.innerHTML = this.value;
        }
    }

    // 4.1 Map Concurrency Slider
    const concurrencySlider = document.getElementById('max-concurrency');
    const concurrencyOutput = document.getElementById('concurrency-val');
    if (concurrencySlider && concurrencyOutput) {
        concurrencySlider.oninput = function() {
            concurrencyOutput.innerHTML = this.value;
        }
    }

    // 5. File Upload Drag & Drop
    const dropZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');

    if (dropZone && fileInput) {
        dropZone.addEventListener('click', () => fileInput.click());

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
        });

        dropZone.addEventListener('drop', handleDrop, false);
        fileInput.addEventListener('change', (e) => handleFiles(e.target.files));
    }

    const selectAllYears = document.getElementById('select-all-years');
    const yearOptions = document.getElementById('year-options');
    const confirmYearsBtn = document.getElementById('confirm-years-btn');
    if (selectAllYears && yearOptions && confirmYearsBtn) {
        selectAllYears.addEventListener('change', () => {
            yearOptions.querySelectorAll('.year-checkbox').forEach((checkbox) => {
                checkbox.checked = selectAllYears.checked;
            });
            updateYearSelectionState();
        });
        yearOptions.addEventListener('change', updateYearSelectionState);
        confirmYearsBtn.addEventListener('click', startSelectedAnalysis);
    }

    // 6. Tutorial Modal (Markdown)
    const tutorialOutput = document.getElementById('tutorial-output');
    
    // Initial content
    const defaultTutorial = `# 📖 食用指南 
> 请详细阅读后进行使用。 
## 这是什么？ 

这是依赖于***Github***上的开源项目***qq-chat-exporter-master***搭建的**QQ**群聊消息总结分析工具，<br>出于**学习科研目的**而开发。 

> 这是***qce***的地址喵~ 
> * \`https://github.com/shuakami/qq-chat-exporter\`  

本工具全部由**谷歌**研发的***Gemini-3-pro-preview***实现，创作环境为**字节跳动**旗下的***trae IDE***。<br>是**货真价实的AIGC**哦，因为作者本人是**代码苦手**的说~ 

此外，本工具调用时可能会把聊天记录完整上传至模型服务提供商，请衡量好**隐私与便利**之间的关系后酌情使用本工具。**若有隐私泄露问题请自行承担责任。** 

## 以下是具体的食用教程： 

### API配置： 
负责配置需要调用到的大语言模型，需要用户自行配置模型服务商的**URL地址**、**API key**、**模型名称**。<br>*推荐使用Gemini-3-pro-preview和Gemini-3-flash。* 

同时提供了单次分析时可以自由选择模型的选项，以便更好的进行分析。**若不需要该功能，请留空即可。** 

### 分析参数： 
**Token预算**：取决于你所配置的模型的上下文长度，你填入的模型的上下文长度越大，那么可使用的预算就可以 *手动* 调至更多。例如： 
> ***Gemini-3-pro-preview***的上下文长度是 *1M* ，那么你就可以将token预算拉满到 *900k* （剩下的100k预算留给提示词与冗余空间）<br>而***官方Deepseek-v3.2***的上下文长度是 *128k* ，那么你就可以将token预算拉到 *120k* <br>（同理，8k作为冗余）

##### *预算上限暂不支持grok的2M上下文。 

当单次输入超过当前接口容量时，工具会自动按聊天记录分段发送并汇总结果。

**动漫小剧场主题**：可以选择预设的两种主题，也可以自定义主题。选择后可以在报告上生成一段将群友带入该主题角色的小剧场，*纯私货，纯整活，ooc可能性存微。* 

**最终输出增强**：开启后在报告生成完毕后再度调用一次LLM，将生成的报告发给LLM将其进行可能存在的**HTML格式修复**与**CSS深度美化**。 



### 历史记录： 
可以在这里查看所有生成过的分析报告，并提供下载服务。**请注意，所有的分析报告是缓存在本地的。** 

### 使用流程： 

**第一步：导出数据** 
<br>使用 ***QQChatExporter (v5)*** 导出 *JSON* 格式记录。该步骤具体教程参见开头提到的网址。 

**第二步：配置 AI** 
<br>点击左上角“API 配置”，填入自定义的LLM API配置，以及开启需要的功能。 

**第三步：上传并选择年份**
<br>将 JSON 文件拖入中央区域，先选择需要分析的一个或多个年份，再点击开始分析。

**第四步：获取报告** 
<br>进度条走完后，按选择的模式下载各年份报告或多年集合报告。

### 感谢你看到这里，也感谢使用本工具喵~<br>若觉得用着顺手记得点赞转发哦，若有建议和意见欢迎联系开发者反馈喵。`;

    if (tutorialOutput) {
        // Initial render
        if (typeof marked !== 'undefined') {
            tutorialOutput.innerHTML = marked.parse(defaultTutorial);
        } else {
            console.error("marked library not loaded");
            tutorialOutput.innerHTML = defaultTutorial;
        }
    }
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
}

function handleFiles(files) {
    if (files.length > 0) {
        inspectFile(files[0]);
    }
}

// --- Core Logic ---

function getAnalysisConfig() {
    return {
        mode: document.getElementById('llm-mode').value,
        base_url: document.getElementById('api-base').value,
        api_key: document.getElementById('api-key').value,
        model: document.getElementById('model-name').value,
        model_map: document.getElementById('model-map').value,
        model_reduce: document.getElementById('model-reduce').value,
        model_refine: document.getElementById('model-refine').value,
        max_tokens: parseInt(document.getElementById('sampling-strength').value),
        max_concurrency: parseInt(document.getElementById('max-concurrency').value),
        report_mode: document.getElementById('report-mode')?.value || 'per_year',
        anime_theme: document.getElementById('anime-theme').value,
        custom_theme_prompt: document.getElementById('custom-theme-prompt').value,
        enhance_mode: document.getElementById('enhance-mode').checked
    };
}

async function inspectFile(file) {
    if (!file.name.endsWith('.json')) {
        alert("请上传 JSON 文件！");
        return;
    }

    // UI Reset
    const statusContainer = document.getElementById('status-container');
    const progressBar = document.getElementById('progress-fill');
    const statusMsg = document.getElementById('status-msg');
    const percentSpan = document.getElementById('status-percent');
    const logBox = document.getElementById('log-box');
    const resultActions = document.getElementById('result-actions');
    const yearSelection = document.getElementById('year-selection');

    pendingInspectionId = null;
    statusContainer.style.display = 'block';
    resultActions.style.display = 'none';
    yearSelection.style.display = 'none';
    progressBar.style.width = '0%';
    percentSpan.innerText = '0%';
    statusMsg.style.color = '#666';
    logBox.innerHTML = '';

    const formData = new FormData();
    formData.append('file', file);

    try {
        statusMsg.innerText = "正在上传并识别年份...";
        log("系统: 开始上传文件并解析时间分片...");
        
        const response = await fetch('/api/inspect', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        
        if (!response.ok || data.status === 'error') {
            throw new Error(data.message);
        }

        pendingInspectionId = data.inspection_id;
        renderYearSelection(data);
        progressBar.style.width = '100%';
        percentSpan.innerText = '已识别';
        statusMsg.innerText = `已识别 ${data.years.length} 个年份，请选择后开始分析`;
        log(`系统: 已识别 ${data.total_messages} 条消息，发现 ${data.years.length} 个年份`);

    } catch (error) {
        console.error(error);
        statusMsg.innerText = "❌ 发生错误";
        statusMsg.style.color = "red";
        log(`ERROR: ${error.message}`);
    }
}

function renderYearSelection(data) {
    const yearSelection = document.getElementById('year-selection');
    const summary = document.getElementById('year-selection-summary');
    const yearOptions = document.getElementById('year-options');
    const selectAllYears = document.getElementById('select-all-years');

    const years = Array.isArray(data.years) ? data.years : [];
    summary.innerText = `${data.chat_name || '未命名聊天'} · 共 ${Number(data.total_messages || 0).toLocaleString()} 条消息`;
    yearOptions.innerHTML = years.map((item) => {
        const year = Number(item.year);
        const checked = year === Number(data.default_year) ? ' checked' : '';
        const start = String(item.start_time || '').slice(0, 10);
        const end = String(item.end_time || '').slice(0, 10);
        const count = Number(item.message_count || 0).toLocaleString();
        return `
            <div class="year-option">
                <label class="year-option-label">
                    <input type="checkbox" class="year-checkbox" value="${year}"${checked}>
                    <span>${year}</span>
                </label>
                <span class="year-option-meta">${count} 条 · ${start} 至 ${end}</span>
            </div>`;
    }).join('');

    selectAllYears.checked = false;
    document.getElementById('confirm-years-btn').innerText = '▶ 开始 AI 分析';
    yearSelection.style.display = 'block';
    updateYearSelectionState();
}

function updateYearSelectionState() {
    const yearOptions = document.getElementById('year-options');
    const selectAllYears = document.getElementById('select-all-years');
    const confirmYearsBtn = document.getElementById('confirm-years-btn');
    if (!yearOptions || !selectAllYears || !confirmYearsBtn) return;

    const checkboxes = Array.from(yearOptions.querySelectorAll('.year-checkbox'));
    const selectedCount = checkboxes.filter((checkbox) => checkbox.checked).length;
    selectAllYears.checked = checkboxes.length > 0 && selectedCount === checkboxes.length;
    selectAllYears.indeterminate = selectedCount > 0 && selectedCount < checkboxes.length;
    confirmYearsBtn.disabled = selectedCount === 0 || !pendingInspectionId;
}

async function startSelectedAnalysis() {
    const confirmYearsBtn = document.getElementById('confirm-years-btn');
    const statusMsg = document.getElementById('status-msg');
    const progressBar = document.getElementById('progress-fill');
    const percentSpan = document.getElementById('status-percent');
    const yearSelection = document.getElementById('year-selection');
    const selectedYears = Array.from(
        document.querySelectorAll('#year-options .year-checkbox:checked')
    ).map((checkbox) => Number(checkbox.value));

    if (!pendingInspectionId || selectedYears.length === 0) {
        alert('请至少选择一个年份');
        return;
    }

    confirmYearsBtn.disabled = true;
    confirmYearsBtn.innerText = '⏳ 正在启动分析...';
    statusMsg.innerText = '正在启动 AI 分析...';
    statusMsg.style.color = '#666';
    progressBar.style.width = '0%';
    percentSpan.innerText = '0%';

    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                inspection_id: pendingInspectionId,
                selected_years: selectedYears,
                config: getAnalysisConfig()
            })
        });
        const data = await response.json();
        if (!response.ok || data.status === 'error') {
            throw new Error(data.message);
        }

        const taskId = data.task_id;
        pendingInspectionId = null;
        yearSelection.style.display = 'none';
        log(`系统: 任务已创建 [ID: ${taskId}]，年份: ${selectedYears.join('、')}`);
        pollProgress(taskId);
    } catch (error) {
        console.error(error);
        confirmYearsBtn.disabled = false;
        confirmYearsBtn.innerText = '▶ 开始 AI 分析';
        statusMsg.innerText = '❌ 启动失败';
        statusMsg.style.color = 'red';
        log(`ERROR: ${error.message}`);
    }
}

async function pollProgress(taskId) {
    const progressBar = document.getElementById('progress-fill');
    const statusMsg = document.getElementById('status-msg');
    const percentSpan = document.getElementById('status-percent');
    const resultActions = document.getElementById('result-actions');

    const interval = setInterval(async () => {
        try {
            const res = await fetch(`/api/status/${taskId}`);
            const data = await res.json();

            // Update Logs
            if (data.new_logs && data.new_logs.length > 0) {
                data.new_logs.forEach(l => log(l));
            }

            // Update Progress
            const pct = data.progress || 0;
            progressBar.style.width = `${pct}%`;
            percentSpan.innerText = `${pct}%`;
            statusMsg.innerText = data.status_text || "处理中...";

            if (data.state === 'completed') {
                clearInterval(interval);
                statusMsg.innerText = "✅ 分析完成！";
                resultActions.style.display = 'block';
                renderResultLinks(data);
                loadHistory(); // Refresh history
            } else if (data.state === 'failed') {
                clearInterval(interval);
                statusMsg.innerText = "❌ 分析失败";
                statusMsg.style.color = "red";
                log(`ERROR: ${data.error}`);
            }

        } catch (e) {
            console.error("Polling error", e);
        }
    }, 1000);
}

function renderResultLinks(data) {
    const resultLinks = document.getElementById('result-links');
    if (!resultLinks) return;

    let reports = Array.isArray(data.result_urls) ? data.result_urls : [];
    if (reports.length === 0 && data.result_url) {
        reports = [{
            label: '下载 HTML 报告',
            url: data.result_url,
        }];
    }

    resultLinks.innerHTML = '';
    reports.forEach((report) => {
        if (!report || !report.url) return;
        const link = document.createElement('a');
        link.href = report.url;
        link.target = '_blank';
        link.className = 'btn-primary';
        link.innerText = `📥 ${report.label || '下载 HTML 报告'}`;
        resultLinks.appendChild(link);
    });

    if (resultLinks.childElementCount === 0) {
        resultLinks.innerText = '报告文件未找到';
    }
}

function log(msg) {
    const box = document.getElementById('log-box');
    const p = document.createElement('div');
    const time = new Date().toLocaleTimeString();
    p.innerText = `[${time}] ${msg}`;
    box.appendChild(p);
    box.scrollTop = box.scrollHeight;
}

async function loadHistory() {
    try {
        const res = await fetch('/api/history');
        const records = await res.json();
        
        const list = document.getElementById('history-list');
        list.innerHTML = '';
        
        if (records.length === 0) {
            list.innerHTML = '<div style="text-align:center; color:#999;">暂无记录</div>';
            return;
        }
        
        records.forEach(r => {
            const div = document.createElement('div');
            div.className = 'history-item'; // You might need to add this class in CSS if not exists, or inline style
            // Using inline style for simplicity as requested "minimalist"
            div.style.padding = '10px';
            div.style.borderBottom = '1px solid #eee';
            div.style.cursor = 'pointer';
            const yearLabel = r.year ? ` · ${r.year} 年` : '';
            const modeLabel = r.report_mode === 'combined' ? ' · 集合报告' : '';
            
            div.innerHTML = `
                <div style="font-weight:bold; color:#333;">${r.chat_name || '未命名群聊'}${yearLabel}${modeLabel}</div>
                <div style="font-size:0.8rem; color:#666;">${r.timestamp}</div>
                <div style="font-size:0.8rem; color:#999;">消息数: ${r.messages_count}</div>
                <a href="/download/${r.report_path.split('\\').pop().split('/').pop()}" target="_blank" style="font-size:0.8rem; color:var(--primary-color);">查看报告</a>
            `;
            list.appendChild(div);
        });

    } catch (e) {
        console.error("Failed to load history", e);
    }
}

async function loadConfig() {
    try {
        const res = await fetch('/api/config');
        const config = await res.json();
        
        if (config.mode) {
            document.getElementById('llm-mode').value = config.mode;
            toggleCustomConfig();
        }
        if (config.base_url) document.getElementById('api-base').value = config.base_url;
        if (config.api_key) document.getElementById('api-key').value = config.api_key;
        if (config.model) document.getElementById('model-name').value = config.model;
        if (config.model_map) document.getElementById('model-map').value = config.model_map;
        if (config.model_reduce) document.getElementById('model-reduce').value = config.model_reduce;
        if (config.model_refine) document.getElementById('model-refine').value = config.model_refine;
        if (config.max_tokens) {
            document.getElementById('sampling-strength').value = config.max_tokens;
            document.getElementById('token-val').innerText = config.max_tokens;
        }
        if (config.max_concurrency) {
            const concurrency = Math.min(16, Math.max(1, Number(config.max_concurrency)));
            document.getElementById('max-concurrency').value = concurrency;
            document.getElementById('concurrency-val').innerText = concurrency;
        }
        if (config.report_mode) {
            const reportMode = document.getElementById('report-mode');
            if (reportMode && ['per_year', 'combined', 'both'].includes(config.report_mode)) {
                reportMode.value = config.report_mode;
            }
        }
        if (config.anime_theme) {
            document.getElementById('anime-theme').value = config.anime_theme;
            toggleCustomTheme();
        }
        if (config.custom_theme_prompt) {
            document.getElementById('custom-theme-prompt').value = config.custom_theme_prompt;
        }
        if (config.enhance_mode !== undefined) {
            document.getElementById('enhance-mode').checked = config.enhance_mode;
        }

    } catch (e) {
        console.error("Failed to load config", e);
    }
}

async function saveConfig(successMsg = '✅ 配置已保存') {
    const config = getAnalysisConfig();

    try {
        const res = await fetch('/api/config', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        });
        const data = await res.json();
        if (data.status === 'success') {
            alert(successMsg);
        } else {
            alert('❌ 保存失败: ' + data.message);
        }
    } catch (e) {
        alert('❌ 保存失败: ' + e.message);
    }
}

async function testConnection() {
    const btn = document.getElementById('test-conn-btn');
    const originalText = btn.innerText;
    btn.innerText = '⏳ 测试中...';
    btn.disabled = true;

    const config = {
        mode: document.getElementById('llm-mode').value,
        base_url: document.getElementById('api-base').value,
        api_key: document.getElementById('api-key').value,
        model: document.getElementById('model-name').value
    };

    try {
        const res = await fetch('/api/test_connection', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(config)
        });
        const data = await res.json();
        
        if (data.success) {
            alert('✅ ' + data.message);
        } else {
            alert('❌ ' + data.message);
        }
    } catch (e) {
        alert('❌ 网络请求失败: ' + e.message);
    } finally {
        btn.innerText = originalText;
        btn.disabled = false;
    }
}
