# 电商搜索召回

![](docs/0.png)

一个毫无 NLP 经验的人的比赛（挖坑填坑）之旅。https://muyuuuu.github.io/2022/03/24/E-commerce-Search-Recall/

1. 实现 DSSM baseline，直接优化距离结果很差，得分 0.057
2. 实现 CoSENT，余弦距离得分 0.159
3. 实现 SimCSE，得分 0.227

tools 里面是精度转换和结果文件检查。

# 参考

- [CoSENT 实现](https://github.com/shawroad/CoSENT_Pytorch)
- [SimCSE 实现](https://github.com/zhengyanzhao1997/NLP-model/tree/main/model/model/Torch_model/SimCSE-Chinese)