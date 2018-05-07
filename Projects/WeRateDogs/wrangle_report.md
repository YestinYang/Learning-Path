# WeRateDogs数据整理

## 1. 数据收集

### 数据来源1: WeRateDogs推特档案

根据项目要求，从GitHub Repo下载该数据档案到本地后，通过`pandas.read_csv` 读取并保存为`pandas.DataFrame` 。

该档案为2355条推特信息，包含每条推特的唯一id，内容，图片链接，评分，狗名及地位，发推时间和设备，回复和转发状态。

### 数据来源2: 推特图片预测 

根据项目要求，通过`pandas.read_csv` 直接通过url下载并读取该数据档案，并设置参数`sep='\t'` 以将档案正确保存为`pandas.DataFrame` 。

该档案为2075条推特的图片预测信息，包含每条推特的唯一id，图片链接，图片编号，概率最高的三个预测结果、概率和是否为狗。

### 数据来源3: 每条推特的额外数据 

根据Twitter API及Tweepy的数据调取规则，可以通过推特的唯一id获得关于该推特的所有信息。此处尝试了使用Twitter API获取数据，通过设置参数`wait_on_rate_limit=True` 和`wait_on_rate_limit_notify=True` ，经过三轮数据调取，完成了数据来源1中推特的额外数据收集。

但为了项目审查的方便，在后续的数据评估和清理中，依然使用了Udacity提供的现有数据`tweet_json.txt` 。首先通过`with…as…` 和`readline()` 的方式逐行读取txt文件并保存在`list` instance中，再通过`json.loads` 读取`id` 、`retweet_count` 和`favorite_count` ，最终保存为`pandas.DataFrame` 。

## 2. 数据评估

对于每个数据档案，都先通过Excel进行visual assessing，再通过编程进行评估。

### 对于WeRateDogs推特档案`twitter-archive-enhanced.csv` 

- `in_reply_to_status_id` 、`in_reply_to_user_id` 和`retweeted_status_id`、 `retweeted_status_user_id` 、`retweeted_status_timestamp` 的非空行都属于回复或转发的信息。考虑到项目的要求— 只考虑含有图片的原始评级，那么由于回复和转发属于非原始评级，属于数据质量问题。
- `expanded_urls` 包含了每条推特的图片信息，若此项为空值，那么对应行所代表的推特不含有图片。同样依照上述项目要求，应该被删除，属于数据质量问题。并且该列的同一数据格中有重复的URL，需要删除，属于数据整洁问题。
- `source` 的值皆为html链接标签，但从数据分析角度，只有其中的content具有分析价值。此处属于数据质量问题。
- `name` 是从推特的`text` 中提取的宠物名字，但明显的问题在于其中包含了大量不是名字的英文量词或介词，需要去掉。此处属于数据质量问题。
- `rating_denominator` 是评分的分母，但此分母为图片狗数量的倍数，由于缺乏规律，难以进行分析。此处属于数据质量问题。
- `timestamp` 的数据类型为`object` ，但由于其是日期时间，应该为`datetime` ，且所有时间后面都有+0000的时区编码，并无分析意义。此处属于数据质量问题。
- `doggo` `floofer` `pupper` `puppo` 实际上属于一个观察值— 狗的地位。此处属于数据整洁问题。
- 大量无效值使用None表示，并非DataFrame惯常使用的`numpy.nan` 。此处属于数据质量问题。

### 对于推特图片预测`image-predictions.tsv` 

- `p1` `p2` `p3` 中的名字缺乏统一格式，有些首字母大写，有些却是小写。此处属于数据质量问题。
- 由于该表记录来源于每条推特的内容，是对推特狗种的预测，应该属于WeRateDogs推特档案表的一部分。此处属于数据整洁问题。

### 对于每条推特的额外数据`tweet_json.txt` 

- 由于该表记录是每条推特的内容的一部分，应该属于WeRateDogs推特档案表的一部分。此处属于数据整洁问题。

## 3. 数据清理

首先处理数据整洁度的问题：

1. 删除`expanded_urls` 中每个数据格内重复的URL，发现最多剩余的两个URL，且最多只有一个twitter内部链接。故将此列拆分成`twitter_urls`和 `outter_urls` 两列进行保存。
2. 合并`doggo` `floofer` `pupper` `puppo` 四列为一列。
3. 将推特图片预测数据仅提取最高概率且为狗的预测值（包括狗种名称和概率），通过`left join on tweet_id` 合并到主表中。并以同样的方法将额外数据中的`retweet_count` 和`favorite_count` 合并到主表中。

接着处理数据质量的问题：

1. 先进行删除数据条目的部分，即依次删除回复、转发、无图片的推特（行）。
2. 通过正则表达式提取`source` 列中html的content部分，并替换原数据格。
3. 通过正则表达式提取`name` 中以小写字母开头的，或为None的值，并用`numpy.nan` 替换原数据格。
4. 手动处理4行`rating_denominator` 不等于10的倍数的行，接着将该列的所有值除以10并保存为新的表示图片中狗的数量的列`num_of_dog` ，再将`rating_numerator` 除以新保存的列，得到以均值统一化的评分，并保存为新列`avg_rating`  。最后，删除原有的分子和分母列。
5. 通过正则表达式提取`timestamp` 中的+0000，并用空值`''` 替代，以达到删除的目的。
6. 转换`timestamp` 为datetime数据格式。
7. 将所有None全部替换成`numpy.nan` 。
8. 将狗种名称改为首字母大写。