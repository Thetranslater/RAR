# RAR Agent V1 漫画对话提取实现规格

> 文档状态：设计已确认，可进入实现
> 产品名称：RAR Agent（Read And Retrieve）
> 功能范围：从本地图像文件夹提取漫画剧情、角色与对话，生成 DatasetBundle 并导出 ShareGPT 训练数据
> 架构基线：固定 Manga Workflow + 通用模型适配层 + 文件型阶段产物 + 程序化恢复
> 依赖关系：复用现有 Project、Chat、AgentHarness、ModelScheduler、DatasetBundle 和 ShareGPT Exporter

## Problem Statement

RAR 已经具备从文字资源提取角色扮演对话数据的固定 Workflow，但漫画不能可靠地先转换成普通 TextChunk 再套用文字流程。漫画中的对话顺序、说话者、角色身份、章节标题和剧情背景同时依赖页面布局与视觉信息；单独使用传统 CV 或 OCR 只能解决局部识别问题，仍无法完成语义层面的角色归属与剧情理解。

漫画资源通常包含数百至上千张图片。逐页串行调用 VLM 会产生很高的等待成本，而让每个批次依赖前一个批次的角色状态和剧情状态，又会阻止并行处理并放大早期错误。因此，漫画 Workflow 必须让图片批次相互独立地并行提取，再通过文本 LLM 对批次局部结果进行角色名称映射、章节重建和说话者校正。

OCR 可以提高文字识别精度，但它也会识别页码、标题、拟声词和背景文字，不能直接作为最终对白；本地 OCR 还依赖可选的 PaddleOCR、模型文件和 CUDA 环境。OCR 必须是可选增强路径，并且不得阻止未启用 OCR 的漫画提取。

长时间、多批次任务必须支持中断后恢复。RAR 不使用文件哈希判断输入是否发生变化，V1 仅依靠固定产物路径、批次或页面 index，以及空结果占位判断已完成和待恢复单元。SQLite 只维护高层运行状态，不复制每个批次的领域结果。

最终用户仍应从 Project 内的同一 Chat 窗口启动和观察任务。Project 只代表工作目录和安全边界，不代表一个 Dataset；一个 Project 可以拥有多个 Chat、资源、Dataset 和并发提取任务。

## Solution

为 RAR 增加独立的 Manga Workflow。用户选择“漫画/图像”资源类型，提供 Project 内的一个图像文件夹和 Dataset 名称。系统递归扫描支持的图片，按相对路径自然排序，生成稳定的页面表和图片批次；每个批次默认包含五页，并由 VLM 独立、并行分析。

VLM 只负责当前批次的视觉事实：提取说出口的对白、批次局部角色及其外观特征、明确可见的角色名称、显式章节起点，以及包含旁白和内心活动的批次剧情。它不接收上一批剧情或全局角色状态，因此多个批次可以并行运行。批次内角色只使用局部 index；这一 index 不跨批次表达身份。

所有视觉批次完成后，文本 LLM 根据带有明确名称的局部角色观察生成一份紧凑的角色名称参考表。随后，系统把完整参考表和若干批次的局部角色观察交给文本 LLM，将每个局部角色映射为正式角色名称或空值。对话中的局部 speaker index 再确定性地转换为正式名称或 `Unknown`。

章节边界仅来自 VLM 识别的显式视觉标记，例如“第 X 话”或章节标题页。系统按页面位置确定性重建章节；一话对应一个 Plot、一个 PlotChunk，并通常对应一个 Conversation。文本 LLM 以整话剧情、初步对白和角色参考表为输入，再次校正说话者，并可以自由重排、合并、拆分、修正或删除对白。

启用 OCR 时，本地 PaddleOCR 与 VLM 同时处理页面。OCR 文本块被程序化切分为句子，并且只在同页内与 VLM 对白进行一对一模糊匹配；达到阈值的 OCR 句子完整替换 VLM 文本，未匹配 OCR 文本全部丢弃。OCR 不改变对白顺序和 speaker。

Workflow 最终复用统一 DatasetBundle 和 ShareGPT Exporter。漫画 V1 不保存页面级来源引用；为了满足统一领域契约，每条最终 Utterance 只引用所属 PlotChunk。所有中间结果保存在固定名称的文件型产物中，失败单元以 `result: {}` 占位，恢复时只重跑空结果。

## User Stories

1. 作为个人爱好者，我希望从漫画图片生成角色扮演对话训练数据，从而避免逐页人工抄录和整理。
2. 作为用户，我希望漫画提取保持非商用产品定位，同时提供接近成熟生产工具的完整流程。
3. 作为用户，我希望只提供一个本地图像文件夹即可启动任务，从而不必先制作清单文件。
4. 作为用户，我希望输入路径限制在当前 Project 内，从而保持清晰的本地安全边界。
5. 作为用户，我希望 Project 与 Dataset 相互独立，从而在一个工作目录中管理多个漫画和多次提取。
6. 作为用户，我希望从现有 Chat 启动漫画提取，从而让进度、确认和结果报告都回到当前对话。
7. 作为用户，我希望在没有当前 Chat 时自动创建一个提取 Chat，从而始终有任务归属位置。
8. 作为用户，我希望在任务实际运行时禁止该 Chat 输入，从而避免 AgentHarness 与 Workflow 同时修改相同产物。
9. 作为用户，我希望逐阶段确认期间重新获得 Chat 输入能力，从而可以检查和修正阶段结果。
10. 作为用户，我希望自动模式连续完成所有阶段，从而无需等待人工确认。
11. 作为用户，我希望逐阶段模式只在大阶段结束后确认，从而不会为每个页面或批次反复操作。
12. 作为用户，我希望文字提取和漫画提取在 UI 中明确分开，从而知道当前使用哪一种 Workflow。
13. 作为用户，我希望漫画与文字提取共享 Dataset 名称、资源路径、执行模式、Debug、文本模型和导出选项，从而减少重复配置。
14. 作为用户，我希望选择已适配的 VLM，从而能够在不同供应商和模型之间切换。
15. 作为用户，我希望未配置 API Key 的模型显示为不可用并说明所需环境变量，从而快速定位配置问题。
16. 作为用户，我希望 API Key 只从环境变量读取，从而不会被 UI、SQLite、日志或产物保存。
17. 作为用户，我希望默认使用 Qwen3.7 Flash，从而以较低成本获得较强的视觉理解能力。
18. 作为用户，我希望 VLM 缺少密钥时禁止启动而不是偷偷回退，从而明确知道实际使用的模型。
19. 作为用户，我希望文本后处理模型可以不同于 VLM，从而按任务特点选择模型。
20. 作为用户，我希望系统递归读取图片子目录，从而支持常见的分卷和分话文件夹布局。
21. 作为用户，我希望图片按相对路径自然排序，从而让 `2.jpg` 正确排在 `10.jpg` 前面。
22. 作为用户，我希望相对路径保留在页面清单中，从而在同名文件位于不同子目录时仍可区分。
23. 作为用户，我希望系统忽略隐藏目录、隐藏文件和 `__MACOSX`，从而不把系统文件当作漫画页面。
24. 作为用户，我希望扫描器不跟随符号链接，从而不会意外离开输入树或形成循环。
25. 作为用户，我希望 V1 支持 JPG、JPEG、PNG 和 WebP，从而覆盖常见漫画图片格式。
26. 作为用户，我希望任务开始前验证所有图片可读取，从而不会运行到中途才发现损坏页面。
27. 作为用户，我希望存在损坏图片时阻止任务并列出问题，从而先修复输入再消耗模型额度。
28. 作为用户，我希望空图片文件夹阻止任务，从而不会生成没有意义的 Dataset。
29. 作为用户，我希望每个文件夹输入成为一个 Manga DatasetResource，从而在最终结果中保留资源边界。
30. 作为用户，我希望 DatasetResource meta 包含页面数量，从而快速了解输入规模。
31. 作为用户，我希望子目录默认只作为页面 meta，从而不会强行把文件组织方式解释为剧情边界。
32. 作为用户，我希望可以把目录边界配置为硬章节边界，从而适配已经按话整理的资源。
33. 作为用户，我希望图片批次默认包含五页，从而平衡上下文、视觉理解和并行效率。
34. 作为用户，我希望批次大小可在一到十页之间配置，从而适配模型限制和漫画版式。
35. 作为用户，我希望批次不跨越目录边界，从而保留资源整理者提供的局部结构。
36. 作为用户，我希望批次记录实际全局页面 index，从而正确处理短批次和可配置批次大小。
37. 作为用户，我希望 VLM 返回的 `page_index` 表示批次局部页号，从而让提示词和输出保持简洁。
38. 作为用户，我希望局部页号通过批次页面表映射为全局页号，从而不依赖错误的乘法推导。
39. 作为用户，我希望每张输入图像带有明确的 `Page 0`、`Page 1` 标签，从而帮助 VLM 返回正确页号。
40. 作为用户，我希望每个视觉批次完全独立，从而可以并行处理大量漫画页面。
41. 作为用户，我希望 VLM 不接收上一批剧情，从而避免批次间串行依赖。
42. 作为用户，我希望 VLM 不接收不断增长的角色列表，从而避免上下文膨胀和早期身份错误传播。
43. 作为用户，我希望最多并行八个 Workflow 模型调用，从而提升速度并为 Chat 留出共享池容量。
44. 作为用户，我希望 Chat 和 Workflow 共用十二槽模型池，从而防止后台提取完全挤占交互能力。
45. 作为用户，我希望 VLM 只提取实际说出口的对白，从而不把旁白和内心独白混入训练对话。
46. 作为用户，我希望旁白、内心活动和其他剧情信息进入 Plot 摘要，从而保留理解对白所需的背景。
47. 作为用户，我希望 VLM 忽略页码、拟声词和无关背景文字，从而减少伪对白。
48. 作为用户，我希望非剧情页面可以产生合法空结果，从而不会被强行制造对话。
49. 作为用户，我希望合法空视觉结果包含完整的空数组和空 Plot 结构，从而与失败占位区分。
50. 作为用户，我希望每个批次的对白保持 VLM 输出顺序，从而保留模型推断的阅读顺序。
51. 作为用户，我希望角色在每个批次内使用局部连续 index，从而稳定引用批次中的说话者。
52. 作为用户，我希望角色 index 接受整数和纯数字字符串，从而容忍常见模型输出差异。
53. 作为用户，我希望角色名称接受 `unknown`、null 和空数组，从而自然表达无法确定身份。
54. 作为用户，我希望 VLM 在视觉证据明确时输出角色名称，从而为后续名称参考表提供锚点。
55. 作为用户，我希望 VLM 同时输出角色外观和身份特征，从而让文本 LLM 能跨批次匹配角色名称。
56. 作为用户，我希望非法、负数、重复或不存在的局部角色引用触发重试，从而不让坏索引污染后续阶段。
57. 作为用户，我希望字符串 speaker 能在局部名称中唯一匹配时自动转换为 index，从而兼容合理的模型输出。
58. 作为用户，我希望无法匹配或存在歧义的字符串 speaker 触发重试，从而避免静默误配。
59. 作为用户，我希望局部角色按首次出现顺序规整为连续 index，从而简化后续映射。
60. 作为用户，我希望显式的“第 X 话”或标题页成为章节起点，从而按漫画自身结构划分对话。
61. 作为用户，我希望模型不根据普通场景切换猜测章节，从而避免过度分割。
62. 作为用户，我希望章节起点所在整页属于新章节，从而让 V1 的页面归属保持确定性。
63. 作为用户，我希望第一个显式章节前存在对白时生成一个隐式开篇章节，从而不丢失封面后的序章内容。
64. 作为用户，我希望第一个显式章节前没有对白时丢弃该部分，从而不为封面和目录生成空章节。
65. 作为用户，我希望没有任何章节标记时把整个文件夹视为一话，从而仍能生成有效结果。
66. 作为用户，我希望批次跨越章节时，按每句对白的页面 index 精确归属章节，从而不重复对话。
67. 作为用户，我希望跨章节批次的 Plot 摘要可同时进入相邻章节背景，从而接受有限且可控的背景重叠。
68. 作为用户，我希望一话对应一个 Plot 和一个 PlotChunk，从而避免再次进行语义剧情切分。
69. 作为用户，我希望一话通常对应一个 Conversation，从而防止跨越多话形成超长训练对话。
70. 作为用户，我希望超长章节不被自动拆分，从而保持已确认的“一话一剧情”语义。
71. 作为用户，我希望章节超出文本模型上下文时明确停止并要求更换大上下文模型，从而避免静默截断。
72. 作为用户，我希望所有批次完成后生成角色名称参考表，从而统一可确定身份的角色名称。
73. 作为用户，我希望名称参考表只使用名称和角色描述作为输入，从而避免无关对话与剧情占用上下文。
74. 作为用户，我希望模糊称谓如“主人公”“学生”“我”“你”被名称参考表过滤，从而减少错误角色合并。
75. 作为用户，我希望角色名称参考表保留正式名称、别名和视觉描述，从而支持后续局部角色映射。
76. 作为用户，我希望名称参考表一次性查看所有具名角色观察，从而获得全局可用的名称视野。
77. 作为用户，我希望名称参考表输入超出模型上下文时停止，从而不通过截断破坏角色映射。
78. 作为用户，我希望空具名观察生成空参考表而不是失败，从而允许身份不明的漫画完成提取。
79. 作为用户，我希望参考表只做轻量确定性清理，从而避免程序擅自进行语义角色判断。
80. 作为用户，我希望每个批次局部角色映射为参考表名称或空值，从而把局部 speaker 转成一致名称。
81. 作为用户，我希望角色映射调用可按模型上下文把多个完整批次分组，从而兼顾吞吐和输入完整性。
82. 作为用户，我希望一个批次永不被拆开到两个角色映射请求中，从而保持批次局部身份关系。
83. 作为用户，我希望完整名称参考表发送给每个角色映射请求，从而保持分组之间的名称一致性。
84. 作为用户，我希望别名能唯一指向正式名称时自动规整为正式名称，从而接受模型使用别名。
85. 作为用户，我希望多个局部角色可以映射到同一个正式名称，从而表示同一角色在不同画面中的观察。
86. 作为用户，我希望无法确定的局部角色最终变成 `Unknown`，从而不强迫模型编造身份。
87. 作为用户，我希望所有映射到同一角色的描述按批次顺序聚合并精确去重，从而生成稳定角色档案。
88. 作为用户，我希望每个具名角色基于聚合 description 生成一次档案，从而避免重复档案。
89. 作为用户，我希望角色档案生成不重新选择名称，从而把身份映射和档案总结职责分开。
90. 作为用户，我希望匿名局部角色不进入 DatasetBundle characters，从而避免产生大量无意义角色。
91. 作为用户，我希望文本 LLM 根据整话 Plot、对白和角色参考表再次检查 speaker，从而修正视觉模型受画面干扰产生的误判。
92. 作为用户，我希望对白修订模型可以重排、合并、拆分、改写和删除错误对白，从而提高最终训练质量。
93. 作为用户，我希望对白修订模型不能修改 Plot，从而让剧情背景和对话修订保持职责分离。
94. 作为用户，我希望修订后的 speaker 只能是正式角色名或 `Unknown`，从而保持 Dataset 一致性。
95. 作为用户，我希望没有对白的章节仍保留 Plot 但不生成 Conversation，从而保留剧情结构而不制造空训练样本。
96. 作为用户，我希望启用本地 OCR 来提高文字准确率，从而修复 VLM 的错别字。
97. 作为用户，我希望 OCR 是可选功能，从而在没有 GPU 或 OCR 依赖时仍可使用漫画 Workflow。
98. 作为用户，我希望程序启动时检测 OCR 依赖、CUDA 和模型目录，从而在 UI 中提前显示可用状态。
99. 作为用户，我希望 OCR 检测不加载完整模型或运行推理，从而避免启动服务时占用大量显存。
100. 作为用户，我希望 OCR 与 VLM 并行运行，从而避免 OCR 把整个视觉阶段串行化。
101. 作为用户，我希望 OCR 结果按终止标点切成句子并保留普通换行，从而可以与 VLM 对白逐句匹配。
102. 作为用户，我希望 OCR 与 VLM 只在同一页内匹配，从而减少跨页误替换。
103. 作为用户，我希望 OCR 匹配忽略空白、换行和可忽略标点差异，从而提高同一句文字的匹配率。
104. 作为用户，我希望 OCR 使用一对一最高分模糊匹配，从而避免一个 OCR 句子替换多句对白。
105. 作为用户，我希望 OCR 匹配阈值默认是 70/100 且可配置，从而调整纠错力度。
106. 作为用户，我希望匹配成功时完整采用 OCR 文字和标点，从而充分利用 OCR 的识别结果。
107. 作为用户，我希望 OCR 替换不改变 speaker 和对白顺序，从而只修正文字内容。
108. 作为用户，我希望未匹配 OCR 结果被丢弃，从而不把页码、标题、拟声词和背景字加入对话。
109. 作为用户，我希望单页 OCR 失败时回退到 VLM 对白并继续，从而让可选增强不会阻塞主流程。
110. 作为用户，我希望 OCR 模型由独立进程加载一次，从而避免每页或每任务重复初始化。
111. 作为用户，我希望多个 Workflow 共享 OCR Worker 和任务队列，从而控制显存占用。
112. 作为用户，我希望 OCR 默认每次处理四页且可高级配置，从而平衡显存和速度。
113. 作为用户，我希望 OCR Worker 崩溃后只自动重启一次，从而处理偶发错误但避免无限循环。
114. 作为用户，我希望最后一个 OCR 任务结束且队列空闲 60 秒后卸载 Worker，从而释放 CUDA 显存。
115. 作为用户，我希望取消任务时移除该任务尚未执行的 OCR 页面，从而避免继续浪费算力。
116. 作为用户，我希望正在执行的 OCR 批次可以结束但不能写入已取消任务，从而保持共享 Worker 一致性。
117. 作为用户，我希望 Debug 模式只处理前五个图片批次及其 OCR 页面，从而快速跑通真实模型测试。
118. 作为用户，我希望每个独立模型单元失败时保留固定位置的空结果，从而保持顺序和恢复能力。
119. 作为用户，我希望其他批次在单个批次失败后继续，从而最大化保留已完成工作。
120. 作为用户，我希望阶段内所有单元结束后再报告不完整，从而一次看到完整失败集合。
121. 作为用户，我希望恢复时只重跑 `result: {}` 的单元，从而节省模型额度。
122. 作为用户，我希望 V1 不计算文件哈希或自动检测输入变化，从而保持恢复机制简单。
123. 作为用户，我希望重新排序或修改输入图片后的后果由用户自行管理，从而不增加 V1 的版本匹配复杂度。
124. 作为用户，我希望 `finish_reason=length` 直接标记失败并建议降低批次大小，从而不重复发送注定相同的请求。
125. 作为用户，我希望 VLM 默认最多尝试两次，从而限制视觉调用成本。
126. 作为用户，我希望文本结构化阶段默认最多尝试三次，从而容忍偶发格式错误。
127. 作为用户，我希望模型输出经过 JSON 结构和领域规则校验，从而不把无效结果写入下游。
128. 作为用户，我希望视觉思考模式的推理内容不被保存，从而避免无关内容进入产物或日志。
129. 作为用户，我希望在供应商提供时记录 reasoning token 数量，从而了解模型用量而不保存思维链。
130. 作为用户，我希望每个阶段产生准确的状态说明，从而知道系统正在扫描、视觉提取还是修订对白。
131. 作为用户，我希望 UI 显示 VLM 批次、OCR 页面、角色映射分组、章节和档案进度，从而观察长任务。
132. 作为用户，我希望最终得到统一 DatasetBundle，从而继续使用已有检查、修改和导出工具。
133. 作为用户，我希望漫画 DatasetBundle 只包含最终角色和对话，从而不复制大量视觉中间数据。
134. 作为用户，我希望漫画 V1 即使不保存页面来源也能让每条 Utterance 引用所属 PlotChunk，从而满足统一领域契约。
135. 作为用户，我希望 DatasetBundle 可以直接导出 ShareGPT，从而立即生成训练数据。
136. 作为用户，我希望 `Unknown` 只作为用户侧上下文而不是目标角色，从而不训练模型扮演未知身份。
137. 作为用户，我希望没有具名角色时允许产生零训练样本并显示警告，从而诚实表达提取结果。
138. 作为用户，我希望得到机器可读的摘要报告，从而快速查看空批次、未知 speaker、OCR 失败和零样本等问题。
139. 作为用户，我希望提示词可替换但输入输出 Schema 固定，从而可以迭代提取效果而不破坏 Workflow。
140. 作为开发者，我希望真实 API 和 CUDA 不成为默认测试依赖，从而稳定运行本地和 CI 测试。

## Implementation Decisions

### 1. Workflow boundary

- Manga Workflow 是与文字 DatasetBuildWorkflow 并列的固定数据生产流程，不把图像转换成 TextChunk，也不在文字 Workflow 中增加大量媒体条件分支。
- Manga Workflow 实现在 Workflow 命名空间内的独立漫画子模块中。内部按编排、领域模型、扫描、规整、角色、章节、OCR 和 OCR Worker 分离职责。
- AgentHarness 不参与批次提取、失败判断或恢复。提取完成后的查询、手工修改、合并和重新导出仍是普通 AgentHarness 工具调用。
- 漫画 Workflow 复用 Project 安全边界、Chat 任务归属、阶段确认、ModelScheduler、模型客户端、DatasetBundle、Artifact 原子写入和 ShareGPT Exporter。

### 2. Input and scanning

- V1 只接收一个 Project 内的本地图像文件夹，不负责联网搜索、下载、压缩包解包或多文件夹组合。
- 扫描器递归收集 JPG、JPEG、PNG 和 WebP，跳过点开头路径、`__MACOSX` 和符号链接。
- 页面以 Project 相对路径自然排序；排序后的零基 index 是整个 Workflow 的全局 `page_index`。
- 每张图片在模型调用前完成存在性、工作目录边界和基本可解码验证。任一损坏文件阻止任务；零图片阻止任务。
- 一个输入文件夹生成一个 `media_type=manga` 的 DatasetResource，其 meta 至少包含页面总数。页面记录可包含相对目录等可扩展 meta。
- 子目录默认不等于章节。配置可以把目录边界提升为硬章节边界。
- V1 不计算输入哈希，不比较文件大小或修改时间，也不自动处理输入变化。

### 3. Image pages and batches

- ImagePage 至少包含全局 `page_index`、Project 相对图片路径和 meta，不使用额外 ID。
- ImageBatch 至少包含连续 `batch_index` 和实际 `page_indexes` 数组。不得通过 `batch_index * batch_size` 推导页面。
- 默认批次大小为五页，允许一至十页。批次不跨目录边界，因此中间可能出现短批次。
- 页面和批次清单在任务开始后固定下来，作为恢复时的单元顺序依据。
- Debug 模式截取前五个 ImageBatch，并只处理这些批次引用的 OCR 页面。

### 4. Provider-neutral multimodal messages

- ModelMessage 的 content 从纯字符串扩展为纯字符串、空值或有序内容块列表。
- V1 内容块包含文本块和本地图像块。图像块只保存经过验证的 Project 相对路径。
- 只有 Manga Workflow 初始使用图像块；AgentHarness 和文字 Workflow 保持文字消息行为。
- ModelRequest 中不保存 base64。模型调度器先授予并发槽，供应商适配器随后惰性读取图片并编码；请求完成后立即释放内存，重试时重新读取。
- 图像 base64 不写入 SQLite、文件产物或日志。
- 通用 OpenAI-compatible 适配器负责把中立内容块转换为供应商消息格式；已验证支持的视觉模型登记为可选模型，而不是假定所有 OpenAI-compatible 模型均支持图像。
- 模型能力包含文本或视觉类型、上下文大小、最大图片数量、可用状态和所需环境变量提示。发起请求前验证批次图片数量不超过模型能力。

### 5. Model registry and selection

- 运行时 ModelRegistry 同时维护 Harness 默认模型、已配置文本模型和已配置视觉模型。
- 模型列表 API 返回可用和不可用模型；不可用项携带缺少的环境变量提示，UI 禁用选择。
- 密钥只从外部环境读取，不通过 UI 输入、不写入配置文件、不写入 SQLite。
- 默认视觉模型为 Qwen3.7 Flash。缺少其密钥时不自动回退；用户必须选择另一个已适配且可用的 VLM。
- 默认文本后处理模型沿用服务或 Harness 当前文本模型，但漫画任务允许显式覆盖。
- 环境配置可以追加模型目录，但只有声明并通过适配能力检查的模型才能出现在对应下拉列表。

### 6. Qwen visual request policy

- 漫画视觉提取默认显式发送 `temperature=1`、`top_p=0.9`、`enable_thinking=true` 和 `reasoning_effort=medium`。
- RAR 通用配置字段保留 `max_output_tokens=16384`；Qwen 适配器将其映射为 `max_completion_tokens=16384`，该预算包含推理和最终回复。
- 默认采用非流式请求，视觉请求超时为 180 秒且可配置。
- 思考内容与 `reasoning_content` 不落盘；若供应商返回 reasoning token 计数，则可写入现有模型用量记录。
- 思考模式下不请求供应商原生 JSON Schema 或 `response_format`。Prompt 要求仅输出 JSON，RAR 提取最终 content 中的 JSON 后执行 Pydantic 和领域校验。
- 这些参数只应用于漫画视觉提取，不改变 Harness、角色参考、角色映射、对白修订或角色档案阶段的默认参数。

### 7. Visual extraction contract

- 每个 ImageBatch 是独立视觉模型单元。输入由可替换系统 Prompt、JSON 任务说明以及按页面顺序交错的页标签和图像组成。
- 视觉模型不接收前一批 Plot、不接收全局角色表，也不读取其他批次结果。
- 输出包含四个部分：`utterances`、`characters`、`chapter_starts` 和 `plot`。
- 每条视觉 Utterance 包含批次局部 `page_index`、局部角色 speaker 或未知值，以及非空对白 content。这里只允许实际说出口的文字。
- 每个局部 Character 包含 batch-local index、名称数组和 description。名称未知时内部统一为空数组；description 用于视觉身份匹配。
- 每个 ChapterStart 包含批次局部 `page_index` 和可空 title；只有明确标题或“第 X 话”等视觉证据才允许输出。
- `plot` 是当前批次的一整段剧情总结，负责容纳旁白、内心活动和非对白剧情。即使一个批次跨越章节也不拆分 plot。
- 非剧情页面可输出空 utterances、characters、chapter_starts 和空 plot；`result: {}` 专用于失败，不能代表合法空内容。
- VLM 负责推断批次内对白阅读顺序；程序保留数组顺序，只在章节组装时按全局页面排序批次之间的内容。

### 8. Visual normalization and validation

- 接受整数 index 和只含数字的字符串 index，并统一转换为整数。
- 接受 `unknown`、null、空字符串或空数组表示未知名称，内部统一为不含 unknown 的 `list[str]`。
- 若 speaker 是局部角色 index，必须存在于当前 characters；若 speaker 是名称字符串，只能在当前角色名称中唯一匹配后转换为 index。
- 负数、非法字符串、重复角色 index、不存在的 speaker、无法匹配或歧义名称均使整个批次结果无效并进入重试。
- 合法局部角色按首次出现顺序重新编号为连续 `0..n-1`，所有 Utterance speaker 同步更新。
- 结构解析、Pydantic 校验和上述领域校验都属于同一次视觉尝试。

### 9. Visual retry and recovery

- 视觉模型单元默认总尝试次数为二，可配置。
- 普通请求错误、无 JSON、Schema 错误或领域校验错误可以进入下一次尝试。
- `finish_reason=length` 不重复相同请求，直接写入空结果，并建议用户降低图片批次大小或提高可用预算。
- 同一阶段内，单个批次失败不取消其他批次；所有批次完成后，若存在空结果则阶段以 incomplete 停止，不执行依赖阶段。
- 视觉产物按 ImageBatch 顺序一行对应一个批次。失败行保留其批次信息和 `result: {}`。
- 恢复时按 batch index 和固定文件位置读取现有结果，只重跑空结果，不依赖 Agent 或 SQLite 推断。

### 10. OCR capability and lifecycle

- OCR 是可选额外依赖。PaddleOCR 文档解析依赖和 Pillow 可由项目 OCR extra 安装；GPU 版 PaddlePaddle 必须由用户按 CUDA 环境单独安装。
- 默认 OCR 模型位于 Project 的模型目录下，并允许通过专用环境变量覆盖。
- 服务启动时通过轻量子进程探测 Python 包、Paddle 版本、CUDA 可用性和模型目录，不加载模型、不执行推理。
- 探测失败时 UI 显示原因并禁用 OCR 开关；漫画主 Workflow 仍可运行。
- OCR Worker 是独立的 Windows spawned process，在第一次 OCR 任务时加载模型，并通过共享队列服务多个 Workflow。
- Worker 内默认 OCR batch 为四页，可作为高级选项配置。
- 系统维护活动 OCR Workflow 引用数。活动数归零且队列为空后等待 60 秒，再终止 Worker 以释放模型和 CUDA 显存。
- Worker 意外退出时自动重启一次；再次失败后相关 OCR 页面记为失败并回退 VLM。
- 取消某 Workflow 时，从队列移除该 Workflow 未执行页面；已经执行中的批次可以结束，但结果写入前必须检查任务是否仍有效。

### 11. OCR artifacts and text preparation

- 启用 OCR 时，每页结果包含 page index、原始 OCR 文本块、可选布局信息和处理状态。
- OCR 文本块不被假定为句子。确定性预处理按终止标点切分，保留普通换行；每个句子继承原始页号和布局位置。
- OCR 页面失败不形成主 Workflow 的空批次，只记录警告并使用该页原始 VLM 对白。
- OCR 和 VLM 同时开始。视觉提取阶段只有在 VLM 批次和当前任务 OCR 页面都完成后才算完成。

### 12. OCR alignment

- OCR Alignment 是纯程序化阶段；OCR 未启用时跳过，并让后续阶段直接读取视觉提取结果。
- 候选匹配严格限制在同一全局页面内。
- 比较前规整空格、普通换行和可忽略标点，但替换结果使用完整 OCR 原文。
- 匹配采用一对一最高分策略，默认 RapidFuzz 阈值 70/100，可配置。
- 达到阈值时只替换 VLM Utterance content，不改变 speaker、page index 或 Utterance 顺序。
- 未匹配 OCR 文本全部丢弃，不进入 Plot 或 Conversation。
- Alignment 产物保持与视觉批次一一对应，供后续角色与章节阶段统一读取。

### 13. Character catalog

- 角色名称参考表在视觉提取和 OCR Alignment 之后、局部角色映射之前生成。
- 输入收集所有名称非空的 batch-local Character observation，只发送 names 和 description，不发送图像、Plot 或对白。
- 此阶段使用一次文本 LLM 调用，不截断、不分层、不递归合并。输入超过所选文本模型上下文时停止并要求更换大上下文模型。
- 输出是紧凑 NamedCharacterCatalog，每项包含正式 name、aliases 和用于视觉匹配的 description。
- Prompt 要求合并别名和重复观察，并过滤“主人公”“叙事者”“我”“你”“学生”“男人”等不稳定泛称。
- V1 只验证 JSON 结构，不验证正式名一定来自输入，也不阻止 LLM 语义上创造、重复或错误合并角色。
- 结果后处理仅执行字符串 trim、移除空值与 unknown、别名精确去重、移除与正式名相同的别名，并按正式名完全相同合并记录、拼接别名和 description。
- 没有任何具名观察时生成空目录并继续；最终所有 speaker 可以为 Unknown，角色档案为空，ShareGPT 可以零样本，同时报告警告。
- 参考表是 Workflow 中间产物，不是持久的全局角色 ID 系统。局部角色身份仍由 `(batch_index, local_character_index)` 表达。

### 14. Character assignment

- Character Assignment 将每个 batch-local Character 映射到参考表中的正式角色名称或 null。
- 输入总是包含完整 NamedCharacterCatalog，并按模型感知的 token 预算装入尽可能多的完整批次角色观察。一个批次永不拆分。
- 多个 Assignment 请求可以并行，最多占用八个 Workflow 模型槽。
- 若完整目录加单个批次仍超出上下文，阶段停止，不截断目录或 description。
- 每个局部角色必须在输出中恰好出现一次。输出名称必须是目录正式名、可唯一解析的 alias 或 null。
- alias 被确定性转换为正式名；歧义 alias、缺项、多项或目录外名称使该 Assignment 单元重试。
- 原始 VLM 名称是强证据但不是锁定值，LLM 可以根据完整参考表和外观描述覆盖。
- 多个局部角色允许映射为同一正式角色。不同真实角色具有相同正式名称是 V1 接受的限制。
- Assignment 记录按分组顺序保存，失败使用 `result: {}`；其他组继续，阶段结束后如有空结果则停止，恢复时只重跑空组。
- 应用映射后，局部 index 转换为正式名称；null 和未确定身份统一为 `Unknown`。

### 15. Chapter reconstruction

- 章节重建是纯程序化阶段，不增加剧情切分 LLM。
- 显式 ChapterStart 转换为全局页面位置并排序；用户启用目录硬边界时，目录首图也形成不可跨越的章节起点。
- ChapterStart 所在整页属于新章节，V1 不尝试把同一页上的对白分配给两个章节。
- 第一个显式起点前若存在对白，生成 title 为 null 的隐式开篇章节；若没有对白则丢弃前置页面剧情。
- 没有任何起点时，整个输入成为一个 title 为 null 的章节。
- Utterance 依据全局 page index 精确归属章节，并在章节内按 page index、批次顺序和 VLM 数组顺序排列。
- 每个 Batch plot 被加入其页面范围覆盖的每个章节。跨章节批次可让同一批次 Plot 背景出现在两个相邻章节，但不得复制 Utterance。
- 每章生成一个 Plot 和一个 PlotChunk。Plot meta 至少包含 chapter index 和可空 title；PlotChunk text 为该章批次 Plot 的顺序拼接，source refs 为空。
- Plot 的 character refs 在角色档案完成后，根据该章对白实际出现的正式角色确定性填充。
- 不做章节语义合并、二次剧情重建或长章节自动切分。

### 16. Dialogue revision

- 每章是一个独立文本 LLM 修订单元，可并行处理，最多使用八个 Workflow 模型槽。
- 输入包含完整章节 Plot、按页面和视觉顺序组装的全部对白，以及 NamedCharacterCatalog 的角色名称信息；不发送图像。
- 模型主要职责是根据文字剧情和完整对话重新判断 speaker，同时允许自由重排、合并、拆分、改写和删除错误对白。
- 输出只包含 utterances；每项含 speaker 和非空 content，不返回 Plot、来源或角色更新。
- speaker 必须是参考表正式名称或 `Unknown`。目录 alias 可在解析时唯一规整为正式名，其余输出无效并重试。
- 合法空 utterances 允许存在，表示该章不生成 Conversation。
- 修订不得修改 Plot。章节 Plot 在此阶段只作为背景输入。
- 单章失败不取消其他章节；失败记录使用 `result: {}`。全部章节处理完后，存在空结果则停止在 Dataset 组装之前；恢复只重跑空章节。
- 若一整章输入不适合所选文本模型上下文，阶段停止并提示选择更大上下文模型，不拆章或截断。

### 17. Character profile generation

- Character Profile 阶段只处理 NamedCharacterCatalog 中最终出现在 Dataset 角色集合内的具名角色。
- 同一正式名称对应的局部 description 按 batch index 和 local index 聚合，完全相同描述精确去重。
- 每个正式角色独立并行生成一次 profile，输入仅包含固定名称、aliases 和聚合 description。
- Profile 模型不能重新选择或改写正式名称。名称身份由 Catalog 与 Assignment 阶段决定。
- 匿名局部角色不生成 CharacterProfile。
- CharacterProfile 的 plot refs 根据修订后 Conversation 中的正式 speaker 确定；没有实际出场引用的目录项不必进入最终 DatasetBundle。
- Profile 单元采用文本结构化阶段的默认三次尝试、空结果占位和仅恢复失败单元规则。

### 18. Dataset and export

- 最终 DatasetBundle 继续使用统一 schema，包含名称、meta、Manga DatasetResource、具名 CharacterProfile 和 Conversation。
- 每个保留章节对应一个 Plot；有非空修订对白时对应一个 Conversation。
- 漫画 V1 不保存页面、图片或 OCR 来源引用。每条最终 Utterance 至少包含所属 PlotChunkRef，以符合统一领域校验；不包含 TextChunkSpanRef。
- `Unknown` 可以出现在 Conversation 中，但不是 CharacterProfile，也不是 ShareGPT 目标角色。
- DatasetBundle 不内嵌页面表、视觉提取、OCR、角色映射或其他中间产物。
- ShareGPT Exporter 直接复用现有按目标角色导出规则，包括 `角色：内容`、Environment 兼容和 assistant loss 标记。
- 如果没有具名角色或没有合格目标回复，允许生成零条 ShareGPT 样本，并在摘要报告中明确警告。

### 19. Stage model

- 漫画 Workflow 的固定顺序是：Image Scan、Visual Extraction、OCR Alignment、Character Catalog、Character Assignment、Chapter Reconstruction、Dialogue Revision、Character Profile、Dataset、Export。
- Visual Extraction 内部同时调度 VLM 批次和 OCR 页面；OCR 未启用时只等待 VLM。
- OCR Alignment 在 OCR 未启用时作为可跳过的程序化阶段。
- Image Scan、OCR Alignment、Chapter Reconstruction、Dataset 和 Export 虽不调用模型，仍发布阶段开始和完成事件。
- 自动模式在所有阶段之间连续执行。逐阶段模式在整个阶段产物完成并写入后暂停确认，不在批次内部暂停。
- 阶段确认控制 Workflow 业务推进，不改变模型 API Key、文件系统或 Tool 权限规则。
- Workflow 运行期间 Chat 输入锁定；等待阶段确认时解锁。确认继续前必须等待该 Chat 中临时 AgentHarness 修改结束。

### 20. Artifact and recovery contract

- Dataset 根目录保存 Workflow 配置、InputManifest、DatasetBundle、导出结果和摘要报告；漫画工作产物集中在独立漫画命名空间内。
- 固定逻辑产物包括：页面清单、批次清单、视觉提取 JSONL、可选 OCR JSONL、可选对齐 JSONL、角色参考表、角色映射 JSONL、Plots 文档、对白修订 JSONL 和角色档案 JSONL。
- 建议的持久化名称分别为 `workflow_config.json`、`input_manifest.json`、`image_pages.jsonl`、`image_batches.jsonl`、`visual_extractions.jsonl`、`ocr_results.jsonl`、`aligned_extractions.jsonl`、`character_catalog.json`、`character_assignments.jsonl`、`plots.json`、`dialogue_revisions.jsonl`、`character_profiles.jsonl`、`dataset.json` 和 `manga_extraction_summary.json`。
- JSON 和 JSONL 写入保持原子性。JSONL 中每个预期单元都有稳定顺序；失败单元保留标识和空 result。
- Workflow 配置保存非敏感的 provider、model、模型参数、批次、OCR 和执行配置，使重启后可恢复一致调用；绝不保存 API Key。
- 恢复器从固定文件判断最后可继续阶段，并从 JSONL 空结果判断缺失单元。聊天记录不参与程序化恢复判定。
- SQLite 只记录 Workflow run、Chat 关联、高层状态、当前阶段、结果路径、错误和模型用量；每批次进度从文件重建。
- 不创建 MangaRun 的复杂领域状态机、DatasetRevision、角色快照或额外恢复数据库。

### 21. Summary report and observability

- 摘要报告为机器可读 JSON，不要求 V1 实现 HTML 报告。
- 报告至少包含扫描页面数、批次数、章节数、具名角色数、Conversation 数和 ShareGPT 样本数。
- 警告至少覆盖：未知局部角色、`Unknown` 对白、合法空视觉批次、失败视觉批次、OCR 失败页面、无对白章节、空角色目录和零训练样本。
- 报告不提供置信度分数，不把模糊匹配分数解释为总体质量。
- UI 进度在视觉阶段分别显示 VLM batches x/y 与 OCR pages x/y，随后显示角色映射分组、对白修订章节和角色档案进度。

### 22. API, Harness tool and CLI

- 提取 API 使用统一入口和 `workflow_type` 判别字段；文字与漫画配置分别由类型化子结构校验，避免一个无约束参数集合。
- Manga 请求包含通用提取配置，并增加 VLM provider/model、图片批次大小、OCR 开关、OCR batch 和模糊匹配阈值。
- 资源预览 API 接收 Project 相对目录，返回有效图片数量、排序后的开头若干路径、跳过项和验证错误，不上传文件。
- Model API 暴露文本/视觉能力、可用状态和环境变量提示。
- AgentHarness 保留文字提取 Tool，并新增独立 Manga Extraction Tool；不把两种 Workflow 合并成一个巨大 union Tool。
- CLI 保留文字提取命令，并新增漫画提取命令；CLI 与 API 调用相同 Workflow 服务和配置校验。

### 23. Prompt contracts

- 漫画视觉提取、角色参考表、角色映射、对白修订和角色档案均使用可替换 Prompt 文件。
- Prompt 文件名是实际 prompt version 标识；不使用模糊的 `v1`、`v2` 字段代替模板身份。
- Prompt 可以修改措辞和策略，但必须遵守当前阶段固定输入输出 schema。Schema 改动属于 Workflow 版本变更。
- 初始实现可以提供满足 schema 的占位 Prompt，之后通过真实漫画测试迭代效果。

### 24. Concurrency and cancellation

- 所有模型调用通过现有全局 ModelScheduler。进程默认总并发为十二，每个 Workflow 默认上限为八。
- 各视觉批次、角色映射组、章节修订和角色档案在各自阶段内并行；有依赖的阶段严格按阶段顺序执行。
- OCR 使用独立进程和队列，不占模型调度器槽位。
- Chat 请求在全局调度中保持可获得槽位；多个 Dataset Workflow 可以同时运行，但每个 Workflow 最多占八槽。
- 取消 Workflow 时取消尚未开始的模型单元、停止后续阶段并保持已经原子写入的产物，以便用户决定是否恢复。

## Testing Decisions

### Highest-level seam

- 最高测试接缝是完整 Manga Workflow：以临时 Project 中的本地图像文件夹为输入，使用 Scripted ModelClient 和 Fake OCR，执行所有阶段并断言最终 DatasetBundle、Plots、摘要报告与 ShareGPT 导出。
- 该接缝应覆盖自动模式和逐阶段模式，并证明任务进度、Chat 锁定、阶段暂停和继续行为。
- 测试只断言用户可观察的产物、API 响应、错误语义和恢复行为，不锁定私有函数或具体类调用顺序。

### Deterministic component tests

- Scanner 测试覆盖递归扫描、扩展名、自然排序、相对路径、隐藏目录、`__MACOSX`、符号链接、空目录和损坏图片。
- Batching 测试覆盖默认与可配置 batch、短批次、目录边界、全局 page index，以及通过 `page_indexes` 数组完成局部到全局映射。
- Multimodal adapter 测试覆盖文本/本地图像块序列、工作目录边界、调度后惰性 base64、请求后释放、不落盘，以及供应商 payload。
- VLM parser 测试覆盖整数与数字字符串 index、unknown/null/空名称、字符串 speaker 唯一解析、连续 index 规整、合法空结果和非法引用重试。
- Structured response 测试覆盖 Markdown fence、前后说明文字中的 JSON 提取、Pydantic 错误、领域错误、尝试次数和 `finish_reason=length` 行为。
- OCR preparation 测试覆盖块到句子切分、标点、换行、页号和布局信息继承。
- OCR alignment 测试覆盖同页限制、归一化、一对一最高分、阈值、完整替换、未匹配丢弃和失败页回退。
- Character Catalog 测试覆盖空目录、unknown 清理、精确 alias 去重、alias 等于正式名、同名记录合并和上下文超限。
- Character Assignment 测试覆盖完整目录重复输入、完整批次分组、单批超限、exactly-once、alias 规整、null、同名多映射和空结果恢复。
- Chapter Reconstruction 测试覆盖显式起点、目录硬边界、隐式开篇、前置无对白丢弃、无标记整册、起点页归新章、跨章 Plot 共享和 Utterance 不重复。
- Dialogue Revision 测试覆盖正式名/Unknown 校验、合法空章、自由重排结果保留、Plot 不变、上下文超限和单章失败占位。
- Character Profile 测试覆盖 description 顺序聚合、精确去重、匿名排除、每角色一次生成和 plot refs 派生。
- Dataset 测试覆盖 Manga resource、一话一 Plot/PlotChunk/Conversation、空对话章省略 Conversation、仅 PlotChunkRef 和无页面来源引用。
- ShareGPT 测试复用现有 Exporter 用例，并增加 Unknown 不作为目标角色、零样本和漫画 Plot 背景输入。
- Summary 测试覆盖所有计数和警告类别，不测试不存在的置信度。

### Recovery and failure tests

- 恢复测试预先写入部分成功和 `result: {}` 产物，断言只调用缺失的视觉批次、角色映射组、章节或角色档案。
- 单元失败测试断言同阶段其他并行单元继续，阶段结束后才抛出 IncompleteStageError，且后续依赖阶段不运行。
- OCR Worker 测试使用假子进程覆盖单次加载、共享队列、重启一次、取消隔离、活动引用计数和空闲退出，不加载真实 Paddle 模型。
- 取消测试断言已写入产物保留、未执行工作停止、正在执行 OCR 不写入取消任务。

### API and UI tests

- API 测试覆盖 `workflow_type=text|manga` 判别、漫画参数验证、模型能力列表、资源预览、任务状态、阶段确认和同 Chat 冲突。
- UI 测试覆盖资源类型选择、共享与漫画专属字段、不可用 VLM/OCR 禁用原因、路径预览、进度显示、运行时输入锁定及确认时解锁。
- CLI 测试覆盖漫画命令参数转换、Project 相对路径验证和与 API 相同的错误语义。

### Optional integration tests

- 默认测试套件不调用真实 API、不要求网络、不要求 CUDA、不加载 1.9GB OCR 模型。
- 真实 Qwen 视觉测试只有在显式环境开关和 API Key 存在时运行，验证五页输入、思考参数、最终 JSON 解析、reasoning token 记录和超时设置。
- 真实 PaddleOCR 测试只有在显式环境开关、依赖、CUDA 和模型目录均可用时运行。
- 真实集成测试不作为普通 CI 的通过条件；测试样例和成本由执行者明确控制。

### Prior art

- 复用现有文字 Workflow 的 StageCallback、IncompleteStageError、JSONL 占位、文件优先恢复和 ScriptedModelClient 测试模式。
- 复用现有 ModelScheduler 的十二槽共享池和单 Workflow 八槽限制测试。
- 复用现有 DatasetBundle、Plot、PlotChunk、Conversation、CharacterProfile 和 ShareGPT Exporter 的领域测试。
- 参考项目内现有 PaddleOCR 脚本的模型加载与结果访问方式，但通过 Worker 生命周期和可选依赖重新封装。

## Out of Scope

- 网络搜索、网络下载、仅提供作品名称的漫画资源发现。
- ZIP、RAR、CBZ、CBR、PDF、Word、视频或纯音频输入。
- 多个图片文件夹合并为一次 Manga Extraction。
- 自动旋转、裁边、去噪、超分辨率、气泡检测、人物检测或其他传统 CV 预处理。
- 使用 OCR 自动增加 VLM 未提取的对白；未匹配 OCR 结果始终丢弃。
- 拟声词、页码、标题和背景文字作为 Conversation Utterance。
- 页面内章节边界和同一页跨两话的精细区域归属。
- 根据剧情语义自动识别场景边界；V1 只按漫画“话”划分。
- 自动拆分超长章节或超长 Conversation。
- 持久全局角色 ID、跨 Dataset 角色实体、未命名角色的跨批次聚类。
- 角色目录输出的事实性名称验证和防止模型利用常识补全身份。
- 两个不同角色同名时的自动消歧。
- 页面、气泡、边界框或原图来源的可追溯引用。
- 输入哈希、修改时间、文件大小或内容变化检测。
- 多语言 Dataset 版本及翻译。
- HTML 可视化报告和网页内逐句编辑器。
- 新训练格式；V1 漫画流程只复用 ShareGPT。
- 远程部署、云端 OCR 服务、用户账户和多人协作。
- 默认测试中的真实模型质量指标、准确率基准和置信度评分。

## Further Notes

### Confirmed end-to-end flow

用户选择 Manga Workflow 并提供本地图像目录后，系统先生成稳定页面与批次清单。VLM 与可选 OCR 并行处理；OCR Alignment 只修正文案。文本 LLM 随后生成角色名称参考表，将批次局部角色映射为正式名称，再由程序按显式话标题重建章节。每章对话经过文本 LLM 的 speaker 与内容修订，具名角色根据聚合视觉描述生成档案，最后组装统一 DatasetBundle 并导出 ShareGPT。

### File-first recovery principle

固定产物是 Manga Workflow 的事实来源。SQLite 只回答“哪个 Chat 有一个运行到了哪个阶段的任务”，具体完成了哪些页面、批次、角色映射和章节由 JSON/JSONL 文件回答。`result: {}` 是失败占位，不是合法空提取；合法空结果必须符合该阶段完整 schema。

### Identity principle

视觉阶段只承诺批次局部身份。NamedCharacterCatalog 提供正式名称参考，Character Assignment 把局部观察映射到参考名称，但 V1 不创建跨 Dataset 的实体 ID，也不保证所有匿名观察被统一。这个边界保留了并行能力，并把不可靠的视觉身份判断限制在可检查的中间产物中。

### Provenance trade-off

漫画 V1 优先验证提取质量和完整 Workflow，因此暂不保存页面级来源。最终 Utterance 只引用 PlotChunk。后续若增加可视化审核，应基于现有 page index 和批次产物新增漫画专用来源类型，而不是伪造 TextChunkSpanRef。

### Initial implementation order

1. 扩展模型契约、模型能力目录和 OpenAI-compatible 多模态适配。
2. 实现扫描、页面与批次领域模型，以及对应预览 API。
3. 实现视觉 Prompt、输出解析、规整、失败占位和恢复。
4. 实现角色目录、角色映射、章节重建和对白修订。
5. 实现角色档案、DatasetBundle 组装、摘要报告和 ShareGPT 导出。
6. 接入统一 API、Harness Tool、CLI、阶段状态和 Web UI。
7. 实现可选 OCR 能力探测、Worker、结果持久化和 Alignment。
8. 通过脚本模型与 Fake OCR 完成端到端测试，再逐步启用受环境控制的真实 Qwen 和 PaddleOCR 测试。

### Implementation readiness criterion

当用户能够在 Project Chat 中选择漫画提取、预览并确认本地图像排序、使用默认五页批次并行调用可配置 VLM、选择性启用本地 OCR、在中断后只恢复失败单元、生成角色参考与局部角色映射、按显式“话”边界得到一话一 Plot 和修订 Conversation、生成具名角色档案、组装有效 DatasetBundle、导出 ShareGPT，并从摘要报告看到所有重要降级和警告时，本规格视为完成实现。
