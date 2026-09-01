# 剧情数据构建

## 1. 小说切块

在项目根目录运行：

```powershell
conda activate rlff-dev
python src/script/build/chunker.py `
  "D:\BaiduNetdiskDownload\小说\1002狐笛的彼方.txt" `
  "artifacts/plot/狐笛的彼方.chunks.json" `
  --token 4096
```

默认卷规则采用宽松匹配：只要某个独立行以“第一卷”“第二部”“第三篇”等卷标记
开头，就把它视为卷标题行。章节规则通过 `--chapter-splitter` 指定，章节也必须是
独立行并以章节模板开头。对于“第一部 第一话 樱花抄”这种格式，会先在卷标题行内
查找章节标记；对于“第一卷”后面另起一行的“第一章 标题”，则会在卷内容中查找。
同一卷中重复出现的卷前缀会合并处理，不会把每个章节误判成新卷。
卷标题和章节标题都可以没有标题内容，只要标题标记被识别即可。如果一个卷内完全
没有识别到章节，就直接把整卷作为一个 section，再按完整句子累计。一个 chunk 在
加入最后一句后达到或略微超过 `--token`，不会为满足预算而截断句子。每个输出 chunk
仍只记录 `volume`、`chapter`、实际 token 数和正文；其在数组中的位置就是原文顺序。
例如“第二卷 第三话 雨夜”会得到 `volume="第二卷"`、`chapter="第三话 雨夜"`。
token 统计默认使用 `o200k_base`，也可以通过 `--token-encoding` 覆盖。

小说编码默认自动尝试 UTF-8 和 GB18030。站点名称、目录等首个章节之前的文本默认丢弃；
需要保留时使用 `--include-front-matter`。

自定义卷、章节标题标记可以分别重复传入：

```powershell
python src/script/build/chunker.py input.txt output.json `
  --volume-splitter "第{num}卷" `
  --chapter-splitter "第{num}章"
```

`{num}` 同时匹配中文数字和阿拉伯数字。复杂格式可使用 `re:` 后跟完整行正则，例如
`--chapter-splitter "re:CHAPTER\s+\d+.*"`。如果未传入
`--volume-splitter`，卷规则默认为以 `第…卷/部/篇` 开头；如果未传入
`--chapter-splitter`，则不进行章节切分，只按卷回退。

### 只检测卷标题

如果想先检查一本小说会识别出哪些卷标题，而不切分正文：

```powershell
python src/script/build/detect_volume_titles.py `
  "D:\BaiduNetdiskDownload\小说\1002狐笛的彼方.txt"
```

默认每行输出一个标题。添加 `--line-numbers` 可同时显示原文行号；添加
`--output detected_titles.txt` 可将结果写入 UTF-8 文本文件。

## 2. 调用 LLM 提取剧情

剧情提取器使用 `src/LLM/create_llm.ts`，支持 DeepSeek、GLM 和 Qwen。在 `.env`
中配置相应 API key、`RLFF_PLOT_PROVIDER` 和 `RLFF_PLOT_MODEL`，然后运行：

```powershell
node --experimental-strip-types src/script/build/extract_plots.ts `
  "dataset/extraction/chunk/1文学少女.json" `
  "dataset/extraction/plot/1文学少女.jsonl" `
  --limit 20 `
  --context-sentences 3 `
  --concurrency 1 `
  --stream
```

也可以继续使用 `python src/script/build/extract_plots.py ...`，该文件会启动同目录的
TypeScript 实现。找不到 Node 时可通过 `RLFF_NODE_BINARY` 指定 Node 可执行文件。

`plot_extraction.txt` 中独立的 `----------` 是消息分隔符：之前的内容作为不变的
system message，之后的 `===输入===` 与每个 chunk 对应的 `input`、`previous`、`next`
JSON 组成 user message，从而保留稳定前缀供服务端 prompt cache 使用。`volume` 和
`chapter` 仅作为输出元数据以及相邻关系判断，不会混入待提取的 `input` 正文。相邻上下文
只在同一卷标题段内生成，避免把标题两侧误认为连续正文。

`plots` 中的每个边界可以是原文字符串或 `null`。当剧情的起始句或尾句位于相邻 chunk
时，对应边界使用 `null`；两个边界不能同时为 `null`，缺失边界也不使用空字符串代替。
同一次调用还会输出当前 chunk 中识别到的 `characters`，每个角色包含 `name` 别名数组
和 `description`。角色结果与剧情边界一起原样保存在该 chunk 的 JSONL `result` 中。

输出为可恢复的 JSONL。再次运行会跳过已经成功写入的 `chunk_id`；使用 `--overwrite`
可从头生成。`--limit 20` 表示本次最多处理 20 个尚未完成的 chunk，重复运行会继续下一
批。正式调用前可以用 `--dry-run --limit 1` 检查消息结构而不读取 API key。添加
`--stream` 后使用 provider 的流式接口；流式片段会先被累计，流结束后再执行 JSON 解析、
原文边界校验和 JSONL 写入。

自动重试处理超时、长度截断、非法 JSON 和结果校验错误。一次响应中的全部校验错误会
合并到同一条反馈，并附上上一轮完整输出，要求模型只修复问题而不重新执行任务。若模型
返回 `finish_reason=length`，第一次重试会使用从当前提示词动态生成的 v2 system prompt
（仅删除第 5 条 `[ADVISE]剧情的开头*可能*出现在新自然段。`）；v2 仍然因 length
停止时，下一次尝试继续使用 v2，并将 DeepSeek 的 `reasoning_effort` 从 `max` 调整为
`high`。

添加 `--debug` 后，会为本次实际处理的每个 chunk 写入完整调用日志：

```text
debug/plots/<输入 JSON 文件名>/<chunk-id>-log.txt
```

例如 `debug/plots/1文学少女/chunk-000000-log.txt`。每份日志包含模型及调用参数、
每一次尝试实际发送的完整 messages、thinking、OUTPUT，以及 JSON 解析和结果校验的
状态或错误。OUTPUT 包含模型返回的 `content`、`finish_reason`、输入 token、输出 token
和总 token，便于判断响应是否因长度上限停止。thinking 不再打印到终端，也不会写入结果 JSONL。日志不包含
API key；同一 chunk 重新运行时会覆盖其旧日志。`--debug-every` 和 `--debug N` 的
间隔功能已废除。每次尝试开始时，日志会立即写入 `ATTEMPT`、`started_at` 和本次实际
发送的 messages，以及该次使用的 prompt variant 和 reasoning effort；完成时再写入
`ATTEMPT ... DONE`、`finished_at`、耗时及处理结果。
同时使用 `--debug --stream` 时，thinking 和正文片段会在接收过程中即时追加到该
chunk 的日志文件，而不是等完整响应结束后一次性写入。

## 3. 恢复完整剧情并重新切块

剧情提取完成后，使用原始 chunk JSON 和剧情结果 JSONL 恢复完整剧情，再为状态提取
生成默认512 token的句子完整块：

```powershell
python src/script/build/rebuild_plots.py `
  "dataset/extraction/chunk/1文学少女.json" `
  "dataset/extraction/plots/1文学少女.jsonl" `
  "dataset/extraction/rebuilt/1文学少女.json" `
  --token 512
```

脚本不依赖 JSONL 的写入顺序，而是使用 `chunk_id` 与原始 chunk 数组重新对齐。
`[start,null]` 会开启跨 chunk 剧情，后续 `plots: []` 的 chunk 会被完整追加，直到
首个 `[null,end]` 将剧情关闭。如果续接 chunk 因模型输出不稳定而以 `[start,end]`
开头，脚本会忽略该 `start`，按 `[null,end]` 处理。

如果当前剧情以 `[null,end]` 开头、但上一剧情已经用非空 `end` 结束，脚本不再将其视为
`orphan null start` 错误，而是把上一剧情重新视为 `truncated`：忽略其结束边界并继续拼接
当前剧情。同一源 chunk 内出现 `[start,end]`、`[null,next_end]` 时也采用相同原则，合并为
一个从 `start` 延续到 `next_end` 的较长剧情。只有不存在任何可扩展的前置剧情时才会报错。

所有重组 warning 在输出到终端的同时，也会覆盖写入
`debug/plots/<书名>/rebuild-warnings-log.txt`。日志包含本次输入、输出、warning 总数和
完整 warning 内容；没有 warning 时同样写入日志，并记录 `warning_count: 0`。

每个恢复后的剧情会合并其实际涉及的全部源 chunk 的角色结果。例如剧情跨越
`c1,c2,...,cn`，则其候选角色为对应 `ch1,ch2,...,chn` 的并集。具有任一相同别名的
记录会视为同一角色，别名按首次出现顺序合并；描述不一致时保留信息更完整的较长描述。
同一个源 chunk 中若包含多个剧情，该 chunk 的候选角色会提供给这些剧情，因此结果可能
有少量过包含，但不会漏掉跨 chunk 出现的已知角色。

默认严格要求每个原始 chunk 都有剧情结果、所有边界都能在原文中精确找到，并且文件
结束时不存在未闭合剧情。提取尚未完成时可添加 `--allow-partial`，此时只处理从
`chunk-000000` 开始的最大连续结果前缀，并丢弃末尾未闭合的剧情。

重组时不会在句子之间或原始 chunk 边界额外插入换行。重新分块会保留原文全部字符，
并持续加入完整句子，直到当前 chunk 达到或超过 token budget 后才输出，因此触发预算
的整句话属于当前 chunk；只有每个剧情最后一个尾 chunk 可能小于预算。

输出结构为：

```json
{
  "title": "1文学少女",
  "chunking": {
    "token_budget": 512,
    "token_encoding": "o200k_base"
  },
  "plots": [
    {
      "volume": "第一卷",
      "chapter": "渴望死亡的小丑 第一章 远子学姐是美食家",
      "characters": [
        {
          "name": ["井上心叶", "心叶"],
          "description": "圣条学园文艺社成员"
        }
      ],
      "chunks": [
        {
          "text": "剧情文本",
          "token": 128
        }
      ]
    }
  ]
}
```

## 4. 调用 LLM 提取状态

状态提取器读取上一步生成的重组剧情 JSON，并将每个
`plots[plot_index].chunks[chunk_index].text` 作为一个独立任务：

```powershell
node --experimental-strip-types src/script/build/extract_states.ts `
  "dataset/extraction/rebuilt/1文学少女.json" `
  "dataset/extraction/states/1文学少女.jsonl" `
  --start 1 `
  --limit 20 `
  --concurrency 1 `
  --stream
```

也可以使用全局 Bun：

```powershell
bun src/script/build/extract_states.ts `
  "dataset/extraction/rebuilt/1文学少女.json" `
  "dataset/extraction/states/1文学少女.jsonl" `
  --limit 20
```

默认提示词为项目根目录的 `state_extraction.txt`。独立的 `----------` 之前是可缓存的
system message；user message 包含动态的
`{"characters":[...],"prenodes":[],"input":"..."}` JSON。`characters` 直接取自当前
`plot`，传入模型前会将 `name` 裁剪为只包含源数组第一个元素，即状态节点使用的正式
角色名；源剧情 JSON 不会被修改。

状态结果通过校验时会统一位置预设属性：当且仅当 `Entity.attribute` 中的 attribute
完全等于 `位置` 或 `地点` 时，输出名称会改写为 `Entity.location`。其他属性名、实体名、
description 和 source 均不修改；重试修复结果也使用相同名称进行节点匹配。

支持与剧情提取相同的 `--provider`、`--model`、`--base-url`、`--concurrency`、
`--requests-per-second`、`--retries`、`--retry-delay-ms`、`--timeout-ms`、
`--max-tokens`、`--temperature`、`--limit`、`--overwrite`、`--dry-run`、`--debug`
和 `--stream` 参数。默认最大输出为8192 token，温度为0。

`--start` 使用从0开始的绝对剧情下标：`--start 0` 从第一个剧情开始，`--start 1`
跳过第一个剧情并从第二个开始。脚本先应用 `--start`，再排除已经完成的任务，最后由
`--limit` 控制本次最多处理多少个尚未完成的 chunk；`--limit` 不按剧情数量计数。

状态提取前会把同一剧情全部 chunks 的 `token` 相加；总量小于512 token 的剧情会被
整体跳过，恰好512 token 时仍会处理。过滤不会改变原始 `plot_index`，因此 `--start`
仍使用输入 JSON 中从0开始的绝对剧情下标。

同一剧情中的 chunk 存在顺序依赖：第一个 chunk 使用空 `prenodes`。之后的 `prenodes`
会保留此前所有 `.active`、`.location` 预设状态的最新节点，并完整加入紧邻上一 chunk
的 `result.nodes`；更早 chunk 中的其他非预设状态不会继续保留。因此 `--concurrency`
控制同时处理的剧情数量，而不是同一剧情内并行的 chunk 数；一个剧情内部始终按
`chunk_index` 串行处理。如果某个 chunk 失败，同一剧情中排在它后面的已选 chunk 会被
阻止，其他剧情仍可继续。断点续跑到剧情中段时，脚本会从已有 JSONL 恢复此前各 chunk
的状态结果；缺少任一前置结果时不会用不完整的 `prenodes` 继续。

结果 JSONL 使用 `plot_index` 和 `chunk_index` 断点续跑，不重复保存输入原文或 token：

```json
{
  "model": "deepseek-v4-pro",
  "source_title": "1文学少女",
  "plot_index": 0,
  "chunk_index": 0,
  "volume": "第一卷",
  "chapter": "渴望死亡的小丑 第一章 远子学姐是美食家",
  "result": {
    "nodes": []
  }
}
```

输出会验证根对象只能包含 `nodes`，每个节点必须拥有规定字段，`name` 必须符合
`Entity.attribute`。`source` 优先在当前输入原文中精确匹配；精确匹配失败时，会寻找
相似度最高的连续原文片段，相似度达到 80% 即通过，并将 `source` 替换为该实际原文。
`initialization` 不对 `before` 作额外限制；所有节点的 `before` 和 `after` 须为
合法 JSON 标量，两者相同时也允许输出。

超时、长度截断、非法 JSON 和所有结果校验错误会在 `--retries` 范围内重试。一次响应中
发现的字段结构、名称、状态值、`type`、`source` 和描述错误会统一发送，不会逐个消耗
重试次数。下一次请求会在原始输入之后加入上一轮输出和合并后的错误信息，并要求模型只
返回错误 node。存在多个 `source` 错误时，各节点错误依次列出，原文只在反馈末尾附加
一次。修复结果按 `name` 合并：返回已知名称的子集时覆盖这些节点；返回名称
完整覆盖原结果的整体 `nodes` 时只覆盖本轮有错的节点；出现原结果中没有的新名称时，
新节点会追加到结果末尾。顶层直接返回的 `nodes` 数组会自动包装为 `{ "nodes": [...] }`。
未返回或无需修复的原节点保持不变。DeepSeek 长度截断后会将 `reasoning_effort` 从
`max` 调整为 `high`。

添加 `--debug` 后，每个状态任务写入：

```text
debug/states/<输入 JSON 文件名>/plot-XXXXXX-chunk-XXXX-log.txt
```

日志包含每次调用实际发送的 messages、调用参数、thinking、原始输出、停止原因、
token 用量及校验结果。启用 `--stream` 时，thinking 和输出正文会实时追加到对应日志。

## 5. 调用 LLM 提取角色对话

对话提取器读取重组剧情 JSON，默认在同一个 plot 内按顺序合并4个约512-token的源
chunk，形成一个约2000-token的独立输入。合并不会跨越 plot 边界，plot 末尾不足4个的
剩余 chunk 会组成一个较短输入：

```powershell
python src/script/build/extract_conversations.py `
  "dataset/extraction/plots/rebuild/2龙与虎.json" `
  "dataset/extraction/conversation/2龙与虎.jsonl" `
  --chunks-per-input 4 `
  --limit 20 `
  --concurrency 3 `
  --stream
```

默认提示词为项目根目录的 `conversation_extraction.txt`。独立的 `----------`
之前作为可缓存的 system message，之后的 `===输入===` 与
`{"characters":[...],"input":"当前合并输入的原文"}` 组成 user message。`characters`
来自重组剧情 JSON 中所有 `plots[].characters` 的全局合并结果：以 `name[0]` 作为正式
角色名去重，合并别名，并将字段名转换为提示词要求的 `names`。模型不再提取角色，只返回
按原文顺序排列的对话和旁白：

```json
{
  "utterances": [
    {
      "speaker": "Environment",
      "content": "*(大河抬头瞪着龙儿。)*"
    },
    {
      "speaker": "逢坂大河",
      "content": "吵死了。"
    }
  ]
}
```

角色对话 `content` 必须与当前 chunk 中按顺序出现的原文连续内容相符；不能把原文中
不连续的多句话合并。脚本从上一次匹配结尾到当前输入末尾进行定位：如果存在一个或多个
精确结果，固定选择第一次出现的位置并记为100分；不存在精确结果时，使用 RapidFuzz
选择整个剩余范围内分数最高的候选，默认阈值为80。100分时仍使用定位并扩展到完整句子
边界的原文作为 `content`。分数低于100但达到阈值时，模型校对后的 `content` 保持不变，
扩展后的原文写入 `source`，供后续状态定位使用。如果匹配片段的首字符或末字符已经是
换行、标点或省略号等分隔符，则对应方向不再继续扩展，避免吞入相邻句子。相似度只在候选
定位时计算一次，扩展后不会重新评分或改变校验结论。模型应优先使用 `characters.names`
中的已知角色名；角色表未覆盖的
匿名发言者可以使用非空的临时名称，例如 `女学生A`。旁白允许在不添加原文外信息的前提
下简化客观动作和环境，但必须使用 `*(...)*` 格式。

所有合并后的输入按全书顺序获得从0开始的 `input_index`。结果同时保留其来源 plot 和
源 chunk 范围，用于断点续跑与后续定位：

```json
{
  "model": "deepseek-v4-pro",
  "source_title": "2龙与虎",
  "input_index": 0,
  "plot_index": 3,
  "source_chunk_start": 0,
  "source_chunk_end": 3,
  "source_token_count": 2048,
  "volume": "第一卷",
  "chapter": null,
  "result": {
    "utterances": []
  }
}
```

脚本支持与状态提取器相同的 `--provider`、`--model`、`--base-url`、
`--requests-per-second`、`--retries`、`--retry-delay-ms`、`--timeout-ms`、
`--max-tokens`、`--temperature`、`--chunks-per-input`、`--start`、`--limit`、`--overwrite`、
`--dry-run`、`--debug`、`--stream` 和 `--fuzzy-threshold`。环境变量优先使用
`RLFF_CONVERSATION_PROVIDER`、`RLFF_CONVERSATION_MODEL`，未配置时回退到剧情提取器的
provider/model。对话提取器的 `--max-tokens` 默认为25000。

对话提取不存在前后输入的状态依赖，因此 `--concurrency 3` 表示最多同时处理3个合并后
的输入。`--start` 表示从0开始的绝对 `input_index`，而不是 `plot_index` 或 plot 内的
源 chunk 下标；`--limit` 表示本次最多处理多少个尚未完成的合并输入。
`--chunks-per-input` 默认为4，可按模型上下文需要修改。不同于状态提取器，对话提取不会
跳过总 token 少于512的短剧情。

超时、长度截断、非法 JSON 和结果校验错误会在 `--retries` 范围内重试。校验错误包括
字段结构错误、空 speaker、旁白格式错误、对话模糊匹配低于阈值或对话顺序与原文不一致。
一次响应中的全部校验错误会合并为一条反馈，因此不会因同一次输出包含多个错误而连续消耗
重试次数。重试时会附上前一次输出与全部错误原因；原文已存在于会话的首条输入中，不会重复
发送。DeepSeek
长度截断后会将 `reasoning_effort` 从 `max` 调整为 `high`。

添加 `--debug` 后，每个任务写入：

```text
debug/conversations/<输入 JSON 文件名>/input-XXXXXX-plot-XXXXXX-chunks-XXXX-XXXX-log.txt
```

日志记录每次尝试的完整 messages、调用参数、thinking、原始输出、token 用量、停止原因
和校验结果；启用 `--stream` 时 thinking 与正文会实时追加。

已有 SFT `messages` 数据可以迁移并合并到同一个 conversation JSONL。迁移器先把每条
SFT 记录匹配回完整剧情，再将消息拆分到其所在的四块输入；assistant 使用
`--character`，`角色：内容` 保留明确角色，`未知角色N` 转为空字符串。迁移记录不伪造
角色描述，因此 `result.characters=[]`：

```powershell
python src/script/build/merge_sft_conversations.py `
  "dataset/extraction/plots/rebuild/2龙与虎.json" `
  "dataset/sft/v2/train.jsonl" `
  "dataset/extraction/conversation/2龙与虎.jsonl" `
  --backup "dataset/extraction/conversation/2龙与虎.extracted.jsonl"
```

迁移前会备份已有模型提取结果；已有记录与 SFT 迁移结果的 `input_index` 发生重叠时会
停止，不会覆盖其中任意一方。合并结果按 `input_index` 排序。

## 6. 将状态节点整理为剧情 transition

状态整理脚本读取重组剧情 JSON 和状态提取 JSONL，将状态节点按其 `source` 在剧情中的
结束位置分组：

```powershell
python src/script/build/build_state_transitions.py `
  "dataset/extraction/plots/rebuild/1文学少女.json" `
  "dataset/extraction/states/1文学少女.jsonl" `
  "dataset/extraction/transitions/1文学少女.json"
```

每个被处理的剧情必须拥有全部 chunk 的状态结果。脚本先临时使用换行符拼接剧情 chunks，
再在节点所属的原始 chunk 中唯一、精确匹配 `source`。结束位置相同的 instant 节点会
合并；因此当一个 source 是另一个 source 的后缀且两者指向原文中的同一处时，它们属于
同一个 transition，外层保留其中最长的 source。所有 initialization 节点统一放入
第一个 `source=""` 的 transition，并按状态名去重；相同状态出现冲突值时停止处理。

输出不保存完整剧情原文、chunks、token 或模型信息：

```json
{
  "title": "1文学少女",
  "plots": [
    {
      "plot_index": 1,
      "volume": "第一卷",
      "chapter": "渴望死亡的小丑 第一章 远子学姐是美食家",
      "transition": [
        {
          "source": "",
          "nodes": [
            {
              "name": "远子学姐.active",
              "before": null,
              "after": true,
              "type": "initialization",
              "description": "表示远子学姐当前是否能够参与场景并进行互动或对话"
            }
          ]
        }
      ]
    }
  ]
}
```

## 7. 构建带状态校验的 RLFF 对话数据

RLFF 数据构建器直接读取状态提取 JSONL、重组剧情和 conversation JSONL，在内部完成原
transition 脚本负责的状态定位与时间线构建。它只连接状态文件中实际存在的连续 chunk
前缀，不会把尚未提取状态的后续剧情误纳入样本：

```powershell
python src/script/build/build_rlff_dataset.py `
  "dataset/extraction/states/2龙与虎.jsonl" `
  "dataset/extraction/plots/rebuild/2龙与虎.json" `
  "dataset/extraction/conversation/2龙与虎.jsonl" `
  "dataset/rlff/2龙与虎/2龙与虎.json" `
  --log "debug/rlff/2龙与虎.jsonl"
```

conversation JSONL 已通过 `plot_index` 明确关联剧情，脚本不再重新猜测记录属于哪个
plot。每条记录会限制在 `source_chunk_start` 至 `source_chunk_end` 的原文范围内，并按
utterance 顺序定位。`speaker="Environment"` 不参与原文定位，其 `*(旁白)*` 会去掉
外层标记并固定使用 `validation=[]`；即使带有 `source` 也会忽略该字段。其他 speaker
原样写入，空字符串表示角色未知。Environment 名称可以通过
`--environment-character` 修改。

四个位置参数依次是：

1. 状态提取 JSONL：即 `extract_states.ts` 的直接输出，每行通过 `plot_index` 和
   `chunk_index` 对应一个剧情 chunk，节点位于 `result.nodes`。
2. 重组剧情 JSON：顶层包含 `title` 和 `plots`，每个剧情包含 `chunks`，每个 chunk
   至少包含原文 `text`。
3. Conversation JSONL：每行包含显式 `input_index`、`plot_index` 和
   `result.utterances`；每条 utterance 包含 `speaker` 与 `content`。当 `content`
   与原文的模糊匹配分数低于 100（但不低于阈值）时，还会包含 `source`：
   `content` 保留模型校对后的文本，`source` 保存扩展定位得到的原文；后续状态定位优先使用
   `source`。精确匹配时不输出 `source`。
4. 最终 RLFF 输出 JSON。

输出只包含真实提取到的 utterance，不再额外生成空内容的 initialization 消息。每个
initialization 在其所属 chunk 的文本开头生效，因此对该 chunk 内的对话立即可见，但
不会泄漏到此前 chunk。后续 chunk 会继承已经可见的状态；同名 initialization 再次出现
时，会在新 chunk 开头更新当前值。instant 状态仍采用惰性引入：只有当其 source 结尾
已经位于当前消息之前时，状态才进入 validation。

states 允许只覆盖剧情的连续 chunk 前缀。脚本只构建从 chunk 0 开始且已有状态提取结果
的部分；没有任何 states 的剧情不进入输出，超出已覆盖前缀的 conversation 记录会跳过，
此前已经构建的消息仍正常保存。对应情况会分别记录为
`conversation_plot_not_covered`、`conversation_chunk_range_not_covered` 或
`partial_plot_coverage`，不会因剧情状态未全量提取而终止整次构建。

`before` 与当前状态不一致、状态尚不存在但转换提供了非空 before、source 或消息定位
歧义等情况会写入日志；可定位的状态即使 before 异常仍会应用 after。`before` 与
`after` 相同的节点不表示状态变化，会在构建时间线时直接忽略且不记录为错误。

输出结构为：

```json
{
  "title": "2龙与虎",
  "plots": [
    {
      "plot_index": 3,
      "messages": [
        {
          "character": "逢坂大河",
          "system": "",
          "validation": [
            {
              "name": "高须龙儿.location",
              "description": "表示高须龙儿当前所在的地点",
              "value": "二年C班教室"
            }
          ],
          "content": "……撞到人连句道歉都不会说吗……？"
        }
      ]
    }
  ]
}
```

每个非 `Environment` 消息都会额外包含 `system` 字段，用于保存角色策略 system
prompt 的文件路径；构建阶段默认写入空字符串。`Environment` 消息不包含该字段。

## 8. 按角色构建初步 SFT 数据

角色 SFT 构建器按 `plot_index` 合并 conversation JSONL 中被拆分的记录，并按
`source_chunk_start`、`source_chunk_end` 和 `input_index` 恢复原始顺序：

```powershell
python src/script/build/build_role_sft_dataset.py `
  "dataset/extraction/conversation/2龙与虎.jsonl" `
  "dataset/extraction/plot/rebuild/2龙与虎.json" `
  "dataset/extraction/tasks/2龙与虎.jsonl" `
  "dataset/sft/roleplay/2龙与虎" `
  --system-template "sft_system_v1.txt" `
  --system-template "sft_system_v2.txt" `
  --seed 42 `
  --exclude-character "小鹦"
```

脚本按输入顺序增量处理整本书各剧情角色表中的 `name` 数组。新数组与零个现有别名组重叠
时创建新组；只与一个现有组重叠时合并到该组；同时与多个现有组重叠时整条忽略，且不计入
名称频次和角色描述。每组内按名称出现频次从高到低排列，频次最高的名称作为规范名。
conversation 中命中任一别名的 speaker 都会转换为该规范名。名称频次及合并结果会写入
`manifest.json`，便于后续人工校对。

tasks JSONL 通过 `plot_index` 与 rebuild 剧情和 conversation 对齐。`result.plot` 作为纯文本
替换 system 模板中的 `{plot}`；`result.tasks` 会先按角色别名归一化，每条“剧情 × 目标角色”
样本只使用该目标角色的 `values`，并以 ASCII 分号 `;` 连接后替换 `{tasks}`，不会混入其他
角色的任务。没有对应 tasks 记录的剧情会被跳过，不会使用 rebuild 原文作为回退。tasks
记录存在但目标角色没有专属任务时，该角色的 `{tasks}` 为空字符串。模板必须同时包含
`{plot}` 和 `{tasks}`。

脚本根据 rebuild 文件的书名，自动在 `dataset/profile/<书名>` 中查找角色档案；文件名前置
数字会在查找时去除。profile 文件名只要命中角色别名组中的任一名称即可使用；没有对应
档案时，使用 rebuild 剧情 `characters[].description` 中同一别名组内最长的非空描述作为
回退。档案和角色描述都不存在时才使用空字符串。`--exclude-character` 可排除不需要作为
assistant 训练目标的
角色，且参数中的别名也会先转换为规范名。

命中 profile 时，其文件名视为角色的正确名称；频次最高名称仅作为内部角色 ID。最终输出的
`details.character`、system prompt 的 `{character}` 以及 user message 中脚本生成的
`角色名:content` 前缀均使用正确名称。没有 profile 时回退到频次名称。utterance 的
`content` 不执行名称替换，因此角色在实际台词中使用的别名会原样保留。manifest 的
`character_name_groups` 同时记录内部 `canonical` 和最终 `resolved_name`，便于核对。

每个“剧情 × 目标角色”只生成一条样本，并从多个 `--system-template` 中等概率选择一个版本。
模板统一交给 `src/script/system_prompt_renderer.py` 解析和渲染，构建脚本只提供
`character`、`profile`、`plot` 和 `tasks` 参数。`<random>` 支持半角或全角分号分隔选项，
相同 `id` 的块在同一条 system prompt 中复用相同选项序号；`<rearrange>` 使用 `<item>`
声明可随机排列的项目，并支持嵌套。`--seed` 同时控制模板选择、random 和 rearrange，使
输出可以复现。

构建时会删除每条 utterance content 内的 `\r` 和 `\n`。同一 speaker（包括
`Environment`）的连续 utterance 会先合并为一条；随后目标角色形成 assistant message，
其余 speaker 按 `speaker:content` 组织并合并为交替的 user message。目标角色位于剧情开头
时，首条对话允许为 assistant message；尾部连续的 user message 会被删除，保证训练样本
以 assistant message 结束。所有 assistant message 都标记 `loss=true`。空 speaker 不会成为
assistant 目标，其 content 不添加角色名前缀，直接作为不计算 loss 的 user 对话历史保留。

输出使用嵌套格式，`all.jsonl` 保存全部样本，`characters/` 按角色分别保存，
`manifest.json` 记录模板、随机种子、角色样本数、源 chunk 覆盖率和未覆盖位置。输出目录
已存在且非空时默认拒绝执行；确认覆盖生成文件时可添加 `--overwrite`。覆盖时会清理旧的
角色 JSONL，避免被排除角色的文件残留。
