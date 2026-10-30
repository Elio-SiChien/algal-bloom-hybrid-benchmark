# 十水体机理–AI水华预测复现包

英文README为完整说明。推荐Python3.10，先安装requirements-analysis.txt，再运行：

```text
python reproduce.py --mode results
```

程序下载固定Release预测包，逐文件核对哈希，从预测和真实观测重算论文指标、统计表与结果图。输出进入新的outputs目录，不覆盖冻结数据。

安装requirements-training.txt及README指定PyTorch后，可运行：

```text
python reproduce.py --mode smoke
python reproduce.py --mode train --method all --site all
python reproduce.py --mode prepare
```

smoke为四法在巢湖的小预算、训练年伪测试；train为完整正式配置入口；prepare从十站原始观测子集重建日数据。小规模试跑不等于四法十水体完整训练已复现。模型检查点未完整保存，本包不承诺直接加载四法预训练模型。

实验使用2023训练、2024评价，未来天气为事后再分析。评价只使用真实目标日，七日涨跌与浓度误差分开。方法名称、适配差异、预算、数据字段及共享条件见英文说明和docs。本包不包含私人论文、批注、账号或历史试验。
