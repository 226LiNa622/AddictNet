#Pathways analysis by Bayesian network
#@author: Li Na == Sun Yat-sen University
#@Supervisor: Xia Wei == Sun Yat-sen University == xiaw23@mail.sysu.edu.cn

##@@ Import Packages @@##
library(tidyverse)
library(VIM)
library(caret)
library(bnlearn)
library(pROC)
library(openxlsx)
library(readxl)
library(Rgraphviz)
library(smotefamily)
library(HydeNet)

# 1. Load data and ensure it is in data.frame format
df <- read_excel("sampling_DAGs.xlsx")
df <- as.data.frame(df)

# 2. Data Preprocessing:
# 2.1 Define categorical variables
factor_vars <- c("DrugusedAround_Yes", "WorkSpecialPlace_Yes", "Drugused")
df[factor_vars] <- lapply(df[factor_vars], as.factor)

# 2.2 Discretize continuous variables
df <- discretize(df, breaks = 2, method = 'hartemink')

# Low recognition of traditional drugs
df$TradDrug_Recognized <- ifelse(df$TradDrug_Recognized == '[1,4]', 1, 0)
df$TradDrug_Recognized <- as.factor(df$TradDrug_Recognized)

# Low risk awareness
df$Drug_Risk_Awareness <- ifelse(df$Drug_Risk_Awareness == '[12,55]', 1, 0)
df$Drug_Risk_Awareness <- as.factor(df$Drug_Risk_Awareness)

# High indulgence attitude
df$Drug_Indulgence <- ifelse(df$Drug_Indulgence == '(31,105]', 1, 0)
df$Drug_Indulgence <- as.factor(df$Drug_Indulgence)

# Low self-efficacy
df$Drug_Self_Efficacy <- ifelse(df$Drug_Self_Efficacy == '(45,85]', 1, 0)
df$Drug_Self_Efficacy <- as.factor(df$Drug_Self_Efficacy)

# Severe externalizing problems
df$Externalize_Problems <- ifelse(df$Externalize_Problems == '(12,64]', 1, 0)
df$Externalize_Problems <- as.factor(df$Externalize_Problems)

# Extreme sensation seeking
df$Sensation_Seeking <- ifelse(df$Sensation_Seeking == '(22,40]', 1, 0)
df$Sensation_Seeking <- as.factor(df$Sensation_Seeking)

# High childhood adversity
df$ACEs <- ifelse(df$ACEs == '(1,10]', 1, 0)
df$ACEs <- as.factor(df$ACEs)

# 3. Model Training: Bootstrap method to determine pathways
# 3.1 Define ensemble function
ensemble <- function(df, bootstrapn, bootstrapt, bl) {
  output <- data.frame(from = character(), to = character())
  for (seed in 1:bootstrapn) {
    set.seed(seed)
    newdata <- df[sample(1:nrow(df), nrow(df), replace = TRUE), ]
    fit <- mmhc(newdata, blacklist = bl, debug = FALSE)
    arcs <- as.data.frame(fit$arcs)
    output <- rbind(output, arcs)
  }
  output$arc <- paste0(output$from, output$to)
  fre <- as.data.frame(table(output$arc))
  fre <- fre[which(fre$Freq >= bootstrapn * bootstrapt), ]
  output6 <- unique(merge(output, fre, by.x = "arc", by.y = "Var1"))
  return(output6)
}

# 3.2 Use the ensemble function
blacklist1 <- data.frame(
  from = c("DrugusedAround_Yes", "WorkSpecialPlace_Yes", "TradDrug_Recognized", 
           "Drug_Risk_Awareness", "Drug_Indulgence", "Drug_Self_Efficacy", 
           "Externalize_Problems", "Sensation_Seeking"),
  to = "ACEs"
)
blacklist2 <- data.frame(
  from = rep("Drugused", ncol(df) - 1),
  to = names(df)[1:ncol(df) - 1]
)
bl <- rbind(blacklist1, blacklist2)
result <- ensemble(df = df, bootstrapn = 1000, bootstrapt = 0.5, bl = bl)

# 3.3 Build network structure
wl <- as.data.frame(result[, c(2:3)])
wl
bn_liq <- model2network("[ACEs][Drug_Indulgence|Drug_Risk_Awareness][Drug_Risk_Awareness|ACEs][Drug_Self_Efficacy|Drug_Indulgence:Externalize_Problems][Drugused|Drug_Self_Efficacy:DrugusedAround_Yes:TradDrug_Recognized:WorkSpecialPlace_Yes][DrugusedAround_Yes|Externalize_Problems][Externalize_Problems|ACEs:Drug_Indulgence][Sensation_Seeking|ACEs:WorkSpecialPlace_Yes][TradDrug_Recognized|Drug_Indulgence:Drug_Risk_Awareness][WorkSpecialPlace_Yes|ACEs:TradDrug_Recognized]")
graphviz.plot(bn_liq, shape = "rectangle")

# 3.3.1 Reverse arcs
bn_liq <- reverse.arc(bn_liq, from = "Sensation_Seeking", to = "WorkSpecialPlace_Yes", check.cycles = TRUE, check.illegal = TRUE)
bn_liq <- reverse.arc(bn_liq, from = "Externalize_Problems", to = "Drug_Indulgence", check.cycles = TRUE, check.illegal = TRUE)

# 3.3.2 Plot DAG
graphviz.plot(bn_liq)

# 4. Calculate BIC and arc strengths
# 4.1 Calculate BIC
c <- BIC(bn_liq, df)
print(c)

# 4.2 Calculate strength of each arc
arc_strength <- data.frame(arc = character(), strength = numeric())
for (i in 1:nrow(bn_liq$arcs)) {
  arc <- bn_liq$arcs[i, c("from", "to")]
  bn_liq_temp <- drop.arc(bn_liq, from = arc[1], to = arc[2])
  bic_without_arc <- BIC(bn_liq_temp, df)
  strength <- bic_without_arc - c
  arc_strength <- rbind(arc_strength, data.frame(arc = paste(arc[1], "->", arc[2]), strength = strength))
}
write.xlsx(arc_strength, "C:/Users/86198/Desktop/statistic analysis/strength.xlsx")

# 5. Parameter Learning
# 5.1 Define parameter learning function
lr <- function(df, bn_liq, bootstrapn) {
  results <- list()
  for (seed in 1:bootstrapn) {
    set.seed(seed)
    # Bootstrap resampling
    newdata <- df[sample(1:nrow(df), nrow(df), replace = TRUE), ]
    # Fit Bayesian network model
    bn.fitted <- bn.fit(bn_liq, data = newdata, method = "bayes")
    # Get conditional probability table for Drugused node
    CP <- bn.fitted$Drugused
    CPT <- as.data.frame(CP[["prob"]])
    # Store frequencies from CPT
    results[[seed]] <- CPT
  }
  # Calculate mean and 95% confidence intervals
  prob_means <- apply(do.call(cbind, lapply(results, function(res) res$Freq)), 1, mean)
  prob_cis <- t(apply(do.call(cbind, lapply(results, function(res) res$Freq)), 1, quantile, probs = c(0.025, 0.975)))
  # Return list with mean and 95% confidence intervals
  list(
    CI = data.frame(
      Drugused = results[[1]]$Drugused,
      Drug_Self_Efficacy = results[[1]]$Drug_Self_Efficacy,
      DrugusedAround_Yes = results[[1]]$DrugusedAround_Yes,
      TradDrug_Recognized = results[[1]]$TradDrug_Recognized,
      WorkSpecialPlace_Yes = results[[1]]$WorkSpecialPlace_Yes,
      Mean_Freq = prob_means,
      Lower_CI = prob_cis[, 1],
      Upper_CI = prob_cis[, 2]
    )
  )
}

# 5.2 Use the function
result <- lr(df, bn_liq, 1000)
write.xlsx(result[["CI"]], "C:/Users/86198/Desktop/statistic analysis/CPT.xlsx")

# 6. Model Prediction
# 6.1 Use the original model on the development sample
test <- read_excel("C:/Users/86198/Desktop/statistic analysis/Dataset/vali-origin.xlsx")
test <- as.data.frame(test)
test[] <- lapply(test[], as.factor)
pre <- predict(bn.fitted, node = "Drugused", data = as.data.frame(test))
confusionMatrix(test$Drugused, pre)

# 6.2 Simulate 1000 rows of data using the fitted Bayesian model
newmd <- rbn(bn.fitted, 10000)
pre <- predict(bn.fitted, node = "Drugused", data = newmd)
confusionMatrix(newmd$Drugused, pre)