# HW1：基于 NumPy 的 EuroSAT RGB 图像分类

本项目使用手写 MLP 完成 EuroSAT RGB 数据集的 10 类图像分类任务。实现中只使用 NumPy 进行矩阵运算，不使用 PyTorch、TensorFlow、JAX 或自动微分框架。

最终实验统一采用 `16x16` 输入尺寸。原始 RGB 图像会先被缩放到 `16x16`，再展平成一维向量输入 MLP。这样可以降低纯 NumPy 训练和网格搜索的计算量，也能减少高维展平输入带来的过拟合风险。

提交链接：

- GitHub Repo: https://github.com/Zhuyy66/CS60003_HW1
- Model Weights: https://drive.google.com/file/d/1GSGULRiR9a9DTfrtshXUQNUXftqh5as7/view?usp=drive_link

## 主要文件

- `solution.py`：最终模型训练、验证、测试、绘图和简要报告生成脚本。
- `grid_search.py`：NumPy 网格搜索脚本，最终实验使用 `--image-size 16`。
- `visualize_weights.py`：第一层隐藏层权重可视化脚本。
- `consolidate_numpy16_results.py`：合并低学习率和高学习率两部分网格搜索结果。
- `generate_final_report_pdf.py`：将 `final_report.md` 导出为 PDF。
- `final_report.md` / `final_report.pdf`：最终实验报告。
- `grid_search_results_numpy16/`：最终 NumPy-only、`16x16` 输入下的网格搜索结果。

## 运行最终模型

直接运行：

```bash
python solution.py
```

默认配置就是最终选出的最佳 NumPy 模型：

- `image_size = 16`
- `hidden_dim = 512`
- `learning_rate = 0.16`
- `lr_decay = 0.90`
- `weight_decay = 5e-4`
- `epochs = 30`

运行后会在 `outputs/` 目录下生成模型权重、训练曲线、混淆矩阵、错例图和简要报告。

## 运行网格搜索

网格搜索分成两部分：低学习率搜索和高学习率搜索。

低学习率部分：

```bash
python -u grid_search.py --image-size 16 --epochs 30 --output-dir grid_outputs_numpy16_30
```

高学习率部分：

```bash
python -u grid_search.py --image-size 16 --epochs 30 --output-dir grid_outputs_numpy16_highlr_30 --hidden-dims 128 256 512 --learning-rates 0.04 0.08 0.12 0.16 --lr-decays 0.90 0.94 0.97 --weight-decays 0.0 0.0001 0.0005
```

合并两部分网格搜索结果：

```bash
python consolidate_numpy16_results.py
```

合并后的最终结果保存在：

```text
grid_search_results_numpy16/
```

## 生成第一层权重可视化

```bash
python visualize_weights.py --model grid_search_results_numpy16/best_model.npz --output-dir grid_search_results_numpy16
```

该脚本会将第一层权重矩阵恢复成 `16x16x3` 的图像形式，用于观察 MLP 是否学到了颜色偏置、空间纹理或与 Forest/River 等类别相关的模式。

## 最终结果

最佳超参数组合：

- Hidden dimension：`512`
- Learning rate：`0.16`
- Learning-rate decay：`0.90`
- Weight decay：`5e-4`
- Best validation accuracy：`0.6859`
- Test accuracy：`0.6881`
- Test loss：`1.1440`

最终模型权重保存在：

```text
grid_search_results_numpy16/best_model.npz
```

## 重要输出文件

- `grid_search_results_numpy16/grid_results.csv`：所有网格搜索组合的结果。
- `grid_search_results_numpy16/summary.json`：最终结果汇总。
- `grid_search_results_numpy16/best_model.npz`：验证集表现最好的模型权重。
- `grid_search_results_numpy16/learning_curves.png`：训练/验证 loss 和 accuracy 曲线。
- `grid_search_results_numpy16/confusion_matrix.png`：测试集混淆矩阵。
- `grid_search_results_numpy16/error_examples.png`：测试集错例图。
- `grid_search_results_numpy16/first_layer_weight_grid.png`：第一层权重可视化。
- `grid_search_results_numpy16/all_class_related_weights.png`：10 个类别各自相关的第一层隐藏单元权重。
- `grid_search_results_numpy16/forest_river_related_weights.png`：与 Forest/River 输出相关的隐藏单元权重。
- `grid_search_results_numpy16/linearized_class_templates.png`：由 `W1 @ W2[:, class]` 得到的近似类别模板。
- `grid_search_results_numpy16/weight_pattern_stats.csv`：类别模板的颜色通道和空间强度统计。

## 依赖

安装依赖：

```bash
pip install -r requirements.txt
```

`requirements.txt` 中只包含 NumPy、Matplotlib 和 Pillow。
