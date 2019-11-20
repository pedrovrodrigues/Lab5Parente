#### Librarys utilizadas
library(sparklyr)
library(dplyr)
library(caret)
library(C50)
library(rpart)
library(rpart.plot)
library(randomForest)
library(caTools)

### Leitura do arquivo tsv com os dados
customersFull = read.csv("customer.tsv", header = T, sep = "\t")

### Transformando as colunas LTV e MONEY_MONTLY_OVERDRAWN em numeric
customersFull$MONEY_MONTLY_OVERDRAWN <- as.numeric(customersFull$MONEY_MONTLY_OVERDRAWN)
customersFull$LTV <- as.numeric(customersFull$LTV)

### Criando uma tabela temporaria para retirar as colunas 
### que nao sao necessarias para as analises
customers = customersFull
customers[,c("CUSTOMER_ID", "LAST", "FIRST", "STATE", "HAS_CHILDREN", 
             "N_TRANS_ATM", "N_TRANS_TELLER", "N_TRANS_KIOSK", 
             "N_TRANS_WEB_BANK")] <- NULL
customers.train.target <- customers[,c("BUY_INSURANCE", "LTV", "LTV_BIN")]
customers.train <- customers

### Removendo a tabela temporaria
rm(customers)

### Criando a conexao com o Spark
sc = spark_connect(master = 'local')

### Copiando a tabela customers.train (tabela somente com as colunas selecionadas) para o spark
customers.spark <- sdf_copy_to(sc, customers.train, overwrite = T)

### Criando um filtro para selecionar quem nao comprou o seguro
not_buyed_insurance = customers.spark %>% filter(BUY_INSURANCE == 'No')

### Enviando essa tabela para o R
not_buyed_insurance = collect(buyed_insurance)

### Selecao aleatoria de 70% dos dados com o Spark
amostra = customers.train %>% sdf_sample(fraction = 0.7)

### Enviando a selecao para o R
amostra = collect(amostra)
amostra[,c("LTV", "LTV_BIN")] <- NULL


### Variavel auxiliar para selecao aleatoria de 70% dos dados
aux = sample.int(nrow(customers.train))
aux = aux[1:floor(0.7*nrow(customers.train))]

customers.test <- customers.train[-aux,] 
customers.test.target <- customers.train.target[-aux,] 
customers.train <- customers.train[aux,]
customers.train.target <- customers.train.target[aux,]

### Retirando as colunas de LTV e LTV_BIN
customers.train[,c("LTV", "LTV_BIN")] <- NULL

### Construcao da arvore utilizando rpart
fit <- rpart(BUY_INSURANCE~., data = customers.train, method = "anova")
rpart.plot(fit)

### Retirando as colunas de LTV e LTV_BIN
customers.test[,c("LTV", "LTV_BIN")] <- NULL

### Previsao do modelo utilizando os dados de validacao
predfit <- predict(fit, customers.test, type = "class")

### Construcao da tabela de confusao
confusao = table(predfit, customers.test.target$BUY_INSURANCE)

### Calculo da acuracia e da precisao
acuracia = sum(diag(confusao))/sum(confusao)
p0 = nrow(not_buyed_insurance)/nrow(customersFull)
kappa = (acuracia-p0)/(1-p0)
precisao = diag(confusao)[1]/sum(confusao[1,])

######## Teste com Random Forest ########

### Retirando algumas colunas
#customers.train[,c("MONEY_MONTLY_OVERDRAWN","PROFESSION", "LTV", "LTV_BIN")] <- NULL
customers.train[,c("PROFESSION")] <- NULL

summary(customers.train)

### Modelo de Random Forest
rf <- randomForest(BUY_INSURANCE~., data = customers.train)

### Predict on the train test
predfit <- predict(rf, customers.test, type = "class")

### Construcao da tabela de confusao
confusao = table(predfit, customers.test.target$BUY_INSURANCE)

### Calculo da acuracia e da precisao
acuracia = sum(diag(confusao))/sum(confusao)
p0 = nrow(not_buyed_insurance)/nrow(customersFull)
kappa = (acuracia-p0)/(1-p0)
precisao = diag(confusao)[1]/sum(confusao[1,])
