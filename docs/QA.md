# 最终验收记录

完整实验成果对应主提交 `f50775d3dd2770bdf63a5df6125d95586ddba435`，后续提交补充验收记录与结果范围说明，不改变训练算法。

- **新 checkout / 新依赖环境**：复用 AutoDL 原有 PyTorch/CUDA，新建项目依赖 venv。9 个命令通过，耗时58秒；包括锁定依赖安装、editable安装、pip check、6个数值/检查点测试、系统诊断、1200步随机smoke与EGL视频、4096步PPO训练、2回合独立评估、完整图表重建。
- **重建一致性**：新 checkout 从提交的真实CSV重建 summary、standalone HTML 和8张结果图，与交付文件的SHA-256逐一一致。
- **浏览器**：直接以file协议打开，无网络依赖；PPO sliders、比较视角、训练/评估选择、配置选择、hover、拖动缩放/重置、窄屏布局均通过，无JavaScript错误。
- **实验追溯**：23个正常训练实验与1个激进学习率实验，均完成2,000,896 transitions；源码与记录Git revision的blob哈希一致，正式baseline seed42的100k数值前缀与独立pilot一致。
- **评估**：20个最终test seeds与selection seeds不重叠；逐回合return、motion审计的reward分解和最终summary一致，样本SD重新计算一致。
- **视频**：最终视频H.264、480×480、1000帧、50秒；本地checkpoint与视频SHA-256均与summary记录匹配，并检查多个时点的跑动姿态。
- **工程检查**：CSV统一LF换行，修复首轮验收中CRLF被Git识别为尾随空白的问题；未改动任何指标值。Git跟踪文件均小于3MiB，凭据/私钥/模型二进制/视频未进入Git。

机器可读证据：`results/qa/fresh_environment.json`、`regeneration.json`、`tutorial.json`、`artifacts.json`。原始安装与浏览器截图日志保留本地，不直接提交。新环境验证指项目依赖隔离，明确复用了主机预装的PyTorch/CUDA，并非重新安装整个操作系统。

视频与checkpoint在本地和AutoDL保留，GitHub不承载这些二进制；新clone可立即查看曲线与Tutorial，视频需从本地交付复制或按README重建。
