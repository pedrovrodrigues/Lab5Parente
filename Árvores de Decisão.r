# Árvores de decisão
#Alguns pacotes R com árvores de decisão: caret, RWeka, ctree, party, tree, C5.0
###############################################################

# CART (Classification and Regression Trees)

# separando conjuntos de treinamento e de teste
library(caret)
set.seed(100)
train.indices <- createDataPartition(iris$Species, p = 0.7, list = FALSE)
iris.train <- iris[train.indices, ]
iris.test <- iris[-train.indices, ]
summary(iris.train)
summary(iris.test)

library(rpart)
# Treinamento
fit <- rpart(Species ~ ., data=iris.train, method="class")
par(xpd = TRUE)
plot(fit,compress=TRUE,uniform=TRUE, main="AD Iris")
text(fit, use.n=TRUE, all=TRUE, cex=.8)

# Teste
prediction <- predict(fit, iris.test, type = "class")
t <- table(iris.test$Species,prediction)
ac <- sum(diag(t))/sum(t)
ac

# Exercício 1:
#  Repita o exercício anterior, usando 10-fold CV para estimar o desempenho da árvore

# estimando probabilidades
prob <- predict(object=fit, newdata=iris.test, type="prob")
head(prob)
tail(prob)

# Olhando parÃ¢metros
?rpart.control

# Removendo poda
fit <- rpart(Species ~ ., data=iris.train, method="class",control = rpart.control(minbucket = 1,cp=0))
par(xpd = TRUE)
plot(fit,compress=TRUE,uniform=TRUE, main="Classification Tree for Iris (without pruning)")
text(fit, use.n=TRUE, all=TRUE, cex=.8)


# Testando Ã¡rvore sem poda
prediction <- predict(fit, iris.test, type = "class")
t <- table(iris.test$Species,prediction)
ac <- sum(diag(t))/sum(t)
ac

# Em regressÃ£o:
# Removendo Ãºltima coluna, da classe
iris_reg.train <- iris.train[-ncol(iris)] 
iris_reg.test <- iris.test[-ncol(iris)] 

fit <- rpart(Petal.Width ~ ., data=iris_reg.train)
#print(fit)
par(xpd = TRUE)
plot(fit,compress=TRUE,uniform=TRUE, main="Ãrvore de regressÃ£o Iris (Petal Width)")
text(fit, use.n=TRUE, all=TRUE, cex=.8)




# Testando Ã¡rvore de regressÃ£o
p <- predict(object=fit, newdata=iris_reg.test)
# NMSE
sum((p-iris_reg.test$Petal.Width)^2)/sum((iris_reg.test$Petal.Width - mean(iris_reg.test$Petal.Width))^2)
# Visualizando algumas prediÃ§Ãµes em relaÃ§Ã£o ao verdadeiro
head(cbind(p,iris_reg.test$Petal.Width)) 

# Exercício 2:
#  Repita a regressão anterior, mas usando 10-fold cross-validation para estimar o NMSE

#########################################################
# C5.0 De: https://rstudio-pubs-static.s3.amazonaws.com/195428_16074a4e980747c4bc05af6c0bb305a9.html

library(C50)
library(printr)

model <- C5.0(Species ~., data=iris.train,trials=1)
summary(model)
plot(model)


# Teste
results <- predict(object=model, newdata=iris.test, type="class")
t <- table(iris.test$Species,results)
ac <- sum(diag(t))/sum(t)
ac

# Probabilidades
# probabilities
prob <- predict(object=model, newdata=iris.test, type="prob")
head(prob)

# ExtraÃ§Ã£o de regras
model <- C5.0(Species ~., data=iris.train,trials=1,rules=TRUE)
summary(model)
results <- predict(object=model, newdata=iris.test, type="class")
t <- table(iris.test$Species,results)
ac <- sum(diag(t))/sum(t)
ac

#############################################
# Visualização de fronteiras
# Código de: http://michael.hahsler.net/SMU/EMIS7332/R/viz_classifier.html

decisionplot <- function(model, data, class = NULL, predict_type = "class",
                         resolution = 100, showgrid = TRUE, ...) {
  
  if(!is.null(class)) cl <- data[,class] else cl <- 1
  data <- data[,1:2]
  k <- length(unique(cl))
  
  plot(data, col = as.integer(cl)+1L, pch = as.integer(cl)+1L, ...)
  
  # make grid
  r <- sapply(data, range, na.rm = TRUE)
  xs <- seq(r[1,1], r[2,1], length.out = resolution)
  ys <- seq(r[1,2], r[2,2], length.out = resolution)
  g <- cbind(rep(xs, each=resolution), rep(ys, time = resolution))
  colnames(g) <- colnames(r)
  g <- as.data.frame(g)
  
  ### guess how to get class labels from predict
  ### (unfortunately not very consistent between models)
  p <- predict(model, g, type = predict_type)
  if(is.list(p)) p <- p$class
  p <- as.factor(p)
  
  if(showgrid) points(g, col = as.integer(p)+1L, pch = ".")
  
  z <- matrix(as.integer(p), nrow = resolution, byrow = TRUE)
  contour(xs, ys, z, add = TRUE, drawlabels = FALSE,
          lwd = 2, levels = (1:(k-1))+.5)
  
  invisible(z)
}

library("rpart")

x <- iris[1:150, c("Sepal.Length", "Sepal.Width", "Species")]
model <- rpart(Species ~ ., data=x)
decisionplot(model, x, class = "Species", main = "CART")


# Com overfitting (sem poda)
model <- rpart(Species ~ ., data=x,
               control = rpart.control(cp = 0, minsplit = 1))
decisionplot(model, x, class = "Species", main = "CART (overfitting)")


library(C50)
model <- C5.0(Species ~ ., data=x)
decisionplot(model, x, class = "Species", main = "C5.0")


set.seed(1000)

library(mlbench)
x <- mlbench.circle(100)
x <- cbind(as.data.frame(x$x), factor(x$classes))
colnames(x) <- c("x", "y", "class")

library("rpart")
model <- rpart(class ~ ., data=x)
decisionplot(model, x, class = "class", main = "CART")


# Sem poda
model <- rpart(class ~ ., data=x,
               control = rpart.control(cp = 0, minsplit = 1))
decisionplot(model, x, class = "class", main = "CART (overfitting)")


library(C50)
model <- C5.0(class ~ ., data=x)
decisionplot(model, x, class = "class", main = "C5.0")

