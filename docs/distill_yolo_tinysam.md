# DINOv2→YOLOv11n 特征蒸馏与 YOLO+TinySAM 推理说明

## 1. 总损失

- 总损失：
  - $ L = L_{\mathrm{det}} + \lambda_f L_{\mathrm{feat}} + \lambda_a L_{\mathrm{att}} + \lambda_k L_{\mathrm{kl}} $

## 2. 检测损失 $L_{\mathrm{det}}$

- 分类（BCE/CE+LS）：
  - $ L_{\mathrm{cls}} = \frac{1}{N}\sum_{i=1}^{N}\mathrm{BCE}(p_i, y_i) $
- 置信度（obj，BCE）：
  - $ L_{\mathrm{obj}} = \frac{1}{M}\sum_{j=1}^{M}\mathrm{BCE}(q_j, o_j) $
- 边框（CIoU 示例）：
  - $ L_{\mathrm{box}} = \frac{1}{G}\sum_{g=1}^{G}\Big(1 - \mathrm{IoU}(b_g, \hat b_g) + \alpha v\Big) $
  - $ v = \frac{4}{\pi^2}\big(\arctan\tfrac{w}{h}-\arctan\tfrac{\hat w}{\hat h}\big)^2,\ \ \alpha = \tfrac{v}{(1-\mathrm{IoU})+v} $
- 合成：
  - $ L_{\mathrm{det}} = \beta_{\mathrm{cls}}L_{\mathrm{cls}} + \beta_{\mathrm{obj}}L_{\mathrm{obj}} + \beta_{\mathrm{box}}L_{\mathrm{box}} $

## 3. 特征蒸馏 $L_{\mathrm{feat}}$

- 设教师特征 $ T^{(l)} \in \mathbb{R}^{H_l\times W_l\times C_T} $，学生特征 $ S^{(l)} \in \mathbb{R}^{H_l\times W_l\times C_S} $，用投影 $ P: \mathbb{R}^{C_S}\to \mathbb{R}^{C_T} $ 对齐通道：
- MSE：
  - $ L_{\mathrm{feat}}^{\mathrm{MSE}} = \sum_{l\in\mathcal{L}} \tfrac{1}{H_lW_l}\lVert P(S^{(l)}) - T^{(l)} \rVert_2^2 $
- Cosine：
  - $ L_{\mathrm{feat}}^{\mathrm{cos}} = \sum_{l\in\mathcal{L}} \tfrac{1}{H_lW_l}\sum_{x,y}\big(1 - \cos(P(S^{(l)}_{x,y}), T^{(l)}_{x,y})\big) $

## 4. 注意力迁移 $L_{\mathrm{att}}$（可选）

- 基于通道能量的空间注意力：
  - $ A_T^{(l)}(x,y) = \tfrac{\sum_{c}(T_{x,y,c}^{(l)})^2}{\sum_{x',y',c}(T_{x',y',c}^{(l)})^2} $
  - $ A_S^{(l)}(x,y) = \tfrac{\sum_{c}(P(S_{x,y,c}^{(l)}))^2}{\sum_{x',y',c}(P(S_{x',y',c}^{(l)}))^2} $
  - $ L_{\mathrm{att}} = \sum_{l\in\mathcal{L}} \lVert A_S^{(l)} - A_T^{(l)} \rVert_2^2 $

## 5. Logits 蒸馏 $L_{\mathrm{kl}}$（可选）

- 温度 $T$ 下软标签 KL：
  - $ p_T = \mathrm{softmax}(z_T/T),\ p_S = \mathrm{softmax}(z_S/T) $
  - $ L_{\mathrm{kl}} = T^2\, \mathrm{KL}(p_T\,\|\,p_S) = T^2\sum_k p_T^{(k)}\log\tfrac{p_T^{(k)}}{p_S^{(k)}} $

- 权重建议（可调）：$ \lambda_f\in[0.5,2.0],\ \lambda_a\in[0.1,1.0],\ \lambda_k\in[0.1,1.0] $。

## 6. 层级尺寸/维度（符号化）

- 输入 $H\times W$（如 640×640）：

| 模块 | 层级 | 空间尺寸 | 通道数 | 说明 |
|---|---|---|---|---|
| YOLOv11n Backbone | C2 | $\tfrac{H}{4}\times\tfrac{W}{4}$ | $C_2$ | 纹理细节 |
| YOLOv11n Backbone | C3 | $\tfrac{H}{8}\times\tfrac{W}{8}$ | $C_3$ | 低层语义 |
| YOLOv11n Backbone | C4 | $\tfrac{H}{16}\times\tfrac{W}{16}$ | $C_4$ | 中层语义 |
| YOLOv11n Backbone | C5 | $\tfrac{H}{32}\times\tfrac{W}{32}$ | $C_5$ | 高层语义 |
| YOLOv11n Neck (PAFPN) | P3 | $\tfrac{H}{8}\times\tfrac{W}{8}$ | $P_3$ | 融合特征 |
| YOLOv11n Neck (PAFPN) | P4 | $\tfrac{H}{16}\times\tfrac{W}{16}$ | $P_4$ | 融合特征 |
| YOLOv11n Neck (PAFPN) | P5 | $\tfrac{H}{32}\times\tfrac{W}{32}$ | $P_5$ | 融合特征 |
| YOLOv11n Head | Detect | 多尺度 | - | 分类/置信度/回归 |
| DINOv2（ViT） | Patch Tokens | $\tfrac{H}{p}\times\tfrac{W}{p}$ | $D_T$ | patch size p=14/16 |
| 投影/对齐 | Proj(S→T) | 匹配学生网格 | $D_T$ | 1×1 Conv/MLP，必要时插值 |

- 典型示例（640×640）：P3=80×80、P4=40×40、P5=20×20；通道数依实现而定（以 `yolo11n.yaml` 为准）。

## 7. 对齐策略

- 空间对齐：双线性插值将教师 token 网格或学生特征重采样到对方网格。
- 通道对齐：$P$ 将学生通道 $C_S$ 投影到教师通道 $D_T$。
- 分层权重：$ L_{\mathrm{feat}} = \sum_{l} w_l L_{\mathrm{feat}}^{(l)} $，可对不同层赋予不同 $w_l$。

## 8. 推理链（YOLO→TinySAM）

- YOLO 产生 bbox 与 conf；取 bbox 与中心点作为 TinySAM 提示，输出二值 mask；可做小目标移除/孔洞填补与可视化保存。
