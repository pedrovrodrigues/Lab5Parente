customersFull = read.csv("customer.tsv", header = T, sep = "\t")
customers = customersFull
customers[,c("CUSTOMER_ID", "LAST", "FIRST", "STATE", "HAS_CHILDREN", 
             "N_TRANS_ATM", "N_TRANS_TELLER", "N_TRANS_KIOSK", 
             "N_TRANS_WEB_BANK")] <- NULL
customers.train.target <- customers[,c("BUY_INSURANCE", "LTV", "LTV_BIN")]
customers.train <- customers
rm(customers)

library(caret)
library(C50)
library(sparklyr)
library(dplyr)
library(rpart)
library(rpart.plot)

aux = sample.int(nrow(customers.train))
aux = aux[1:floor(0.7*nrow(customers.train))]

customers.test <- customers.train[-aux,] 
customers.test.target <- customers.train.target[-aux,] 
customers.train <- customers.train[aux,]
customers.train.target <- customers.train.target[aux,]

customers.train[,c("LTV", "LTV_BIN")] <- NULL
fit <- rpart(BUY_INSURANCE~., data = customers.train)
rpart.plot(fit)
customers.test[,c("LTV", "LTV_BIN")] <- NULL
predfit <- predict(fit, customers.test, type = "class")
confusao = table(predfit, customers.test.target$BUY_INSURANCE)
acuracia = sum(diag(confusao))/sum(confusao)
precisao = diag(confusao)[1]/sum(confusao[1,])
