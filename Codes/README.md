# Data-Mining-project
This repo store the project of the Data Mining lession

姓名：曹涵         学号：201934705    


项目：使用K-Means、Affinity propagation、Mean-shift、Spectral clustering、Ward hierarchical clustering、Agglomerative clustering、DBSCAN、Gaussian mixtures聚类算法对sklearn.datasets.load_di gits和sklearn.datasets.fetch_20newsgroups数据集进行聚类，并采用–	Normalized Mutual Information (NMI)、
Homogeneity、Completeness三个指标对聚类效果进行评价。

一、实验方法

1.对数据集作预处理
2.调用K-Means、Affinity propagation、Mean-shift、Spectral clustering、Ward hierarchical clustering、Agglomerative clustering、DBSCAN、Gaussian mixtures聚类算法对数据集进行聚类
3.采用Normalized Mutual Information (NMI)、 Homogeneity、Completeness验证聚类效果
 二、实验任务
 对多个聚类算法在load_digits 、fetch_20newsgroups数据集上的聚类效果用NMI、Homo、Coml指标进行评估
 
 三、实验数据
 1.sklearn.datasets.load_digits
 2.sklearn.datasets.fetch_20newsgroups
 
 四、实验步骤
 
 1.数据集load_digits
 （1）从sklearn中加载手写数据集，然后用Scale函数对数据集进行预处理
 （2）保存原有数据集的标签至labels_true
 （3）调用K-Means、Affinity propagation、Mean-shift、Spectral clustering、Ward hierarchical clustering、Agglomerative clustering、DBSCAN、Gaussian mixtures进行聚类
 （4）将聚类后的数据集标签传给labels_pred
 （5）将labels_true、labels_pred作参数,调用Normalized Mutual Information (NMI)、 Homogeneity、Completeness函数对聚类结果进行评估
2.数据集：fetch_20newsgroups
 （1）从sklearn中加载新闻数据集
 （2）将文本转换为向量，并使用Tfid进行处理
 （2）保存原有数据集的标签至labels_true
 （3）调用K-Means、Affinity propagation、Mean-shift、Spectral clustering、Ward hierarchical clustering、Agglomerative clustering、DBSCAN、Gaussian mixtures进行聚类
 （4）将聚类后的数据集标签传给labels_pred
 （5）将labels_true、labels_pred作参数,调用Normalized Mutual Information (NMI)、 Homogeneity、Completeness函数对聚类结果进行评估

五、实验结果
<table class="table table-bordered table-striped table-condensed">
   <tr>
      <td>数据集：load_digits</td>
      <td>K-means</td>
      <td>Affinity propagation</td>
      <td>Mean-shift</td>
      <td>Spectral clustering</td>
      <td>Ward hierarchical clustering</td>
      <td>DBSCAN</td>
      <td>Gaussian mixtures</td>
   </tr>
   <tr>
      <td>NMI</td>
      <td>0.625</td>
      <td>0.59</td>
      <td>0.014</td>
      <td>0.878</td>
      <td>0.241</td>
      <td>0.173</td>
      <td>0.676</td>
   </tr>
   <tr>
      <td>Homogeneity</td>
      <td>0.602</td>
      <td>0.964</td>
      <td>0.007</td>
      <td>0.773</td>
      <td>0.139</td>
      <td>0.122</td>
      <td>0.877</td>
   </tr>
   <tr>
      <td>Completeness</td>
      <td>0.65</td>
      <td>0.425</td>
      <td>0.256</td>
      <td>0.878</td>
      <td>0.893</td>
      <td>0.301</td>
      <td>0.55</td>
   </tr>
   <tr>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
   </tr>
   <tr>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
   </tr>
   <tr>
      <td>数据集：fetch_20newsgroups</td>
      <td>K-means</td>
      <td>Affinity propagation</td>
      <td>Mean-shift</td>
      <td>Spectral clustering</td>
      <td>Agglomerative clustering</td>
      <td>DBSCAN</td>
      <td>Gaussian mixtures</td>
   </tr>
   <tr>
      <td>NMI</td>
      <td>0.67</td>
      <td>0.291</td>
      <td>0.531</td>
      <td>0.557</td>
      <td>0.47</td>
      <td>0.268</td>
      <td>0.272</td>
   </tr>
   <tr>
      <td>Homogeneity</td>
      <td>0.624</td>
      <td>0.95</td>
      <td>0.396</td>
      <td>0.613</td>
      <td>0.427</td>
      <td>0.9</td>
      <td>0.494</td>
   </tr>
   <tr>
      <td>Completeness</td>
      <td>0.724</td>
      <td>0.172</td>
      <td>0.803</td>
      <td>0.511</td>
      <td>0.522</td>
      <td>0.158</td>
      <td>0.188</td>
   </tr>
   <tr>
      <td></td>
   </tr>
</table>

六、实验结果和感想
通过本次实验熟悉了sklearn库中的一些常用模块，它的功能很强大。此外，对于8个聚类算法有了更加深入的理解，实验一开始的时候感觉无从下手，不过经过查阅资料，慢慢的将学到的知识实际呈现了出来。当然，实验还有很多做的不好的地方，比如说Mean_shift聚类算法对于load_digits算法的聚类效果并不好，还需要再通过学习来找出问题的所在。
三个评测指标的学习也丰富了知识。总体来说，本次实验收获颇丰。




							
							






