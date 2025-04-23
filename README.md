## 运行
```
bash scripts/run.sh
```

## 集成

在`benchmarks`下创建一个文件夹，文件夹里创建一个`eval.py`文件。写一个派生类继承BenchmarkEvaluator，主要重写2个函数：
1. prepare_input_item: 输入是datasets的一个item，输出是经过处理后，以system_content, user_content, image格式的tuple，其中前两个为str类型，image为PIL.Image类型
2. evaluate_item: 输入是模型生成的response（str），输出是字典。其中字典中必须包含'correct'项(bool)和'response'项(str)
