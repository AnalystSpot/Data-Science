# Prediction using Unsupervised Learning
### Level : Beginner
## Task :
- For the given Iris dataset find the optimum number of clusters.
- Try to form clusters of flowers using petal width and length features. Drop other 2 features for simplicity.
- Find optimum number of custers using Elbow plot

## IRIS dataset :
 <a href='https://github.com/AnalystSpot/Data-Science/blob/main/Prediction%20using%20Unsupervised%20Learning/Iris.csv'>Dataset</a><br/>
The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in his 1936 paper The use of multiple measurements in taxonomic problems as an example of linear discriminant analysis. It is sometimes called Anderson's Iris data set because Edgar Anderson collected the data to quantify the morphologic variation of Iris flowers of three related species. Two of the three species were collected in the Gaspé Peninsula "all from the same pasture, and picked on the same day and measured at the same time by the same person with the same apparatus".

The data set consists of 50 samples from each of 3 species of Iris.

Species : Iris setosa, Iris virginica and Iris versicolor. 

4 Features were measured from each sample : 
- sepal length (cm)	
- sepal width (cm)	
- petal length (cm)	
- petal width (cm)

Based on the combination of these 4 features, Fisher developed a linear discriminant model to distinguish the species from each other.

<a href='https://en.wikipedia.org/wiki/Iris_flower_data_set'>Know more...</a>

<h3 align='center'>Iris Classification</h3>
<img src='iris1.png' title='Iris Classification'/>

<h3 align='center'>Iris Features & Labels</h3>
<img src='iris features & labels.png' title='Iris Features & Lables'/>

## Steps used :
1. Import
2. Visualize
3. Model creation & SSE / WCSS finding
4. Elbow Plot

* SSE /WCSS : Sum of Squared error / Within Cluster Sum of Squares error
<h3 align='center'>Sum of Squares Error or Within Cluster Sum of Squares error</h3>
<img src='within cluster sum of squares.png' title='within cluster sum of squares'/>

<img src='iris petal classification.png'/><img src='iris sepal classification.png'/>
