#Missing value interpolation by missforest & 
#@author: Li Na == Sun Yat-sen University
#@Supervisor: Xia Wei == Sun Yat-sen University == xiaw23@mail.sysu.edu.cn

install.packages("missForest")
library(missForest)
library(openxlsx)
library(readxl)

#Read data
mydata<-read_excel("Drug_data.xlsx")
mydata<- as.data.frame(mydata)
#Determine the subclass variable
mydata[, c(1:11)] <- lapply(mydata[, c(1:11)], factor)
mydata[, c(21:30)] <- lapply(mydata[, c(21:30)], factor)
#Random seed
set.seed(1314)
summary(mydata)
md.pattern(mydata)
#Using missforest
imp <- missForest(mydata, maxiter = 20, ntree = 100, verbose = TRUE)
#Output data after interpolation and export
data<-imp$ximp
write.xlsx(data, file = "after_missforest.xlsx", rowNames = FALSE)