#LOAN REQUEST CLASSIFICATION

#Libraries and options ####
library(tidyverse)
library(GGally)
library(DataExplorer)

options(scipen=999)


#Data prep ####

##Load and inspect data####
df <- read.csv("./OriginalData/clients.csv", header=TRUE)
summary(data.frame(unclass(df), stringsAsFactors = TRUE))
#factors: month, sex, education,  product_type, having_children_flg, region
  #, family_status, phone_operator, is_client, bad_client_target
#all columns are possible features
#no missing values


##Clean data ####


#rename columns
names(df)[8] <- "have_children"
names(df)[13] <- "prev_client"
names(df)[14] <- "bad_client"




#rename factor levels, absorb unfrequent levels
#education
df$education <- recode(df$education, "Higher education"="Higher",
                       "Incomplete higher education"="HigherDropout",
                       "Incomplete secondary education"="Secondary",
                       "PhD degree"="Higher",
                       "Secondary education"="Secondary",
                       "Secondary special education"="SecondarySpec")

#product_type
prod_types <- as.data.frame(table(as.factor(df$product_type)))
#group those with less than 80 obs:
prod_types_rare <- subset(prod_types, Freq <=80)
#master groups:
  #fashion: Cosmetics and beauty services, jewelry
  #vehicles: Auto, Boats
  #recreational: Tourism, Music, Fishing and hunting supplies, Sporting goods, Audio & Video,
    #Fitness 
  #hardware: Windows & Doors, Construction Materials, Repair Services, Garden equipment
  #necessities: Medical services, Training, Children's goods
df$product_type <- recode(df$product_type, 
                          "Cosmetics and beauty services" = "Fashion",
                          "Jewelry" = "Fashion",
                          "Auto" = "Vehicles",
                          "Boats" = "Vehicles"
                          )

lookup_recreational <- tibble(old=c("Tourism", "Music", "Fishing and hunting supplies",
                                    "Sporting goods", "Audio & Video", "Fitness"),
                              new=c(rep("Recreational", 6)))
df$product_type <- recode(df$product_type, !!!deframe(lookup_recreational))


lookup_hardware <- tibble(old=c("Windows & Doors", "Construction Materials", "Repair Services",
                                "Garden equipment"),
                              new=c(rep("Hardware", 4)))
df$product_type <- recode(df$product_type, !!!deframe(lookup_hardware))

df$product_type <- recode(df$product_type, 
                          "Medical services" = "Necessities",
                          "Training" = "Necessities",
                          "Childen's goods" = "Necessities")

df$product_type <- recode(df$product_type, "Fashion"="Beauty")

df$product_type <- recode(df$product_type, "Household appliances"="Appliances")

rm(prod_types, prod_types_rare, lookup_hardware, lookup_recreational)




#factorize factors. only ordinal: education

#month, sex, product_type, family status, region, phone operator: 
  #factorize, don't change levels
for (i in c(1, 5, 7, 9, 11, 12)){
  df[,i] <- as.factor(df[,i])
}
rm(i)


#education: factorize, order levels
summary(as.factor(df$education))
df$education <- factor(df$education, levels=c("SecondarySpec", "Secondary",
                                              "HigherDropout", "Higher"),
                       ordered=TRUE)


#have children, region, phone_operator, prev_client, bad_client
  #recode values, factorize
for (i in c(8, 13, 14)) {
  df[,i] <- recode(as.character(df[,i]),
                   "0" = "No",
                   "1" = "Yes")
  df[,i] <- as.factor(df[,i])
}
rm(i)




#summary of results
summary(df)






#Exploratory analysis ####  


##Numerics ####
ggpairs(df,
        columns=c("credit_amount", "credit_term", "age", "income", "bad_client"),
        mapping=(aes(color=bad_client, fill=bad_client, alpha=0.5)))
#credit_amount:
  #bad clients have slightly higher median credit amount, but also fewer high outliers
  #0.5 correlation with credit_term
  #very right tailed distribution
#credit_term:
  #considerably higher for bad clients
  #0.5 correlation with credit_amount
  #bit more positively correlated with income for bad clients, compared to all
  #right tailed distribution
#age:
  #considerably lower for bad clients
  #bit more positively correlated with income for bad clients
  #right tailed distribution
#income:
  #lower for bad clients
  #a little correlated with credit amount, 0.373
  #bit more positively correlated with credit_term and age for bad clients
  #very right tailed distribution




##Categoricals ####
plot_bar(df, by="bad_client")
#all categoricals have significant relationship with bad_client

plot_bar(df)
#region 1 and phone operator 4 are very rare




##Interactions ####


#possible interactions:
  #credit_amount X product type
  #age X income
  #age X family_status
  #have_children X family_status


#test interactions:

#credit amount X product_type
ggplot(df, aes(product_type, credit_amount, fill=bad_client)) + 
  geom_boxplot()
#for some product types, higher credit amount is associated much more with bad clients
  #example: high credit amount for clothing: bad
#for some, lower credit amount is associated with good clients
  #example: high credit amount for recreation: good
#add interaction


#age X income
ggplot(df, aes(age, income, color=bad_client)) + geom_point() +
  geom_smooth()
#good clients' income peaks around 30, then declines
#bad clients' income peaks around 45, then declines
#small but significant interaction. add


#age X family_status
ggplot(df, aes(family_status, age, fill=bad_client)) + 
  geom_boxplot()
#for married and another, bad clients are younger
#for unmarried, bad clients are a bit older
#add interaction


#age X have_children
ggplot(df, aes(have_children, age, fill=bad_client)) + 
  geom_boxplot()
#without kids, bad clients are much younger
#with kids, bad and good clients are almost at the same age
#add interaction



#have_children X family_status
plot(x=df$family_status, y=df$bad_client)
plot(x=df$family_status:df$have_children, y=df$bad_client)
#bad clients tend to be married > unmarried > another, regardless of children
#no significant interaction


##Summary of EDA ####

#use all original features
#log transform numeric features
#no strong multicollinearity
#interactions to add:
  #credit_amount X product_type
  #age X income
  #age X family_status
  #age X have_children
  




#Feature engineering ####


##Creating features ####

#payment: amount to be paid in 1 period
df <- df %>% mutate(payment = round(credit_amount / credit_term, 0), .after=credit_term)


#ratio_amount: ratio of amount to be paid and income
df <- df %>% mutate(ratio_amount = round(credit_amount / income, 2), .after=credit_amount)


#ratio_payment: ratio of installment to income
df <- df %>% mutate(ratio_payment = round(payment / income, 2), .after=payment)




##Evaluating features ####
ggpairs(df,
        columns=c("payment", "ratio_amount", "ratio_payment", "bad_client"),
        mapping=(aes(color=bad_client, fill=bad_client, alpha=0.5)))
#payment:
  #lower for bad clients
  #positively correlated  with ratio_amount (0.5) and ratio payment (0.732), for bad clients
  #very strong right tail distribution
#ratio_amount:
  #lower for bad clients
  #strong positive correlation with ratio_payment (0.84), but less so for bad clients (0.56)
  #very strong right tail distribution
#ratio_payment:
  #same as ratio_amount
  #lower for bad clients




##Summary of FeatureEng ####

#keep payment
#keep both ratios, or decide between one, or PCA them?