library(quadprog)
library(zoo)
library(tseries)
library(forecast) 

#wczytanie danych
wynagrodzenie = read.csv2("C:/Users/roksi/OneDrive/Pulpit/wynagrodzenie7.csv", header = FALSE)
zycie = read.csv2("C:/Users/roksi/OneDrive/Pulpit/trwanie_zycia.csv", header = FALSE)
wynagrodzenie = wynagrodzenie$V2
zycie = zycie$V2

plot(wynagrodzenie,zycie,ylab="Przeciêtna d³ugoœæ ¿ycia",xlab="Wynagrodzenie", col="blue")
cor(wynagrodzenie,zycie)

#budowa trzech modeli regresji
reglin=lm(zycie~wynagrodzenie)
summary(reglin)
# R2 = 0.939 
coef(reglin)
## Dostajemy prost¹: y=ax+b, gdzie: b=49.041791479, a=0.006610498 
plot(wynagrodzenie, zycie, ylab="Przeciêtna d³ugoœæ ¿ycia",xlab="Wynagrodzenie", type="p", col="blue")
abline(reglin,  lwd=2, col="red")

# Model 2: Regresja kwadratowa
regkw=lm(zycie~wynagrodzenie+I(wynagrodzenie^2)) 
summary(regkw)
# R2 = 0.9408 
coef(regkw)
# Dostajemy krzyw¹: y=ax^2+bx+c, gdzie: a=6.876339e-07  , b=1.574730e-03, c=5.814047e+01
x=seq(min(wynagrodzenie),max(wynagrodzenie),length=100)
linregkw=coef(regkw)[3]*x^2+coef(regkw)[2]*x+coef(regkw)[1]
plot(wynagrodzenie,zycie,ylab="Przeciêtna d³ugoœæ ¿ycia",xlab="Wynagrodzenie", type="p", col="blue")
lines(x,linregkw,lwd=2, col="red")

# Model 3: Regresja szeœcienna
regsz=lm(zycie~wynagrodzenie+I(wynagrodzenie^2)+I(wynagrodzenie^3))
summary(regsz)
# R2 = 0.9608 
coef(regsz)
# Dostajemy krzyw¹: y=ax^3+bx^2+cx+d, gdzie a=-5.407807e-09  , b=6.067502e-05 , c=-2.182927e-01 , d=3.243326e+02 
linregsz=coef(regsz)[4]*x^3+coef(regsz)[3]*x^2+coef(regsz)[2]*x+coef(regsz)[1]
plot(wynagrodzenie, zycie,ylab="Przeciêtna d³ugoœæ ¿ycia",xlab="Wynagrodzenie", type="p", col="blue")
lines(x,linregsz,lwd=2,col="red")

# Model 4: regresja wielomianowa 4 stopnia
reg4=lm(zycie~wynagrodzenie+I(wynagrodzenie^2)+I(wynagrodzenie^3)+I(wynagrodzenie^4))
summary(reg4)
# R2 = 0.9609
coef(reg4)
# Dostajemy krzyw¹: y=ax^4+bx^3+cx^2+dx+e
linreg4=coef(reg4)[5]*x^4+coef(reg4)[4]*x^3+coef(reg4)[3]*x^2+coef(reg4)[2]*x+coef(reg4)[1]
plot(wynagrodzenie, zycie, ylab="Przeciêtna d³ugoœæ ¿ycia",xlab="Wynagrodzenie", type="p", col="blue")
lines(x,linreg4,lwd=2,col="red")

# Model 5: regresja wielomianowa 5 stopnia
reg5=lm(zycie~wynagrodzenie+I(wynagrodzenie^2)+I(wynagrodzenie^3)+I(wynagrodzenie^4)+I(wynagrodzenie^5))
summary(reg5)
# R2 = 0.9622
coef(reg5)
# Dostajemy krzyw¹ y=ax^5+bx^4+cx^3+dx^2+ex+f
linreg5=coef(reg5)[6]*x^5+coef(reg5)[5]*x^4+coef(reg5)[4]*x^3+coef(reg5)[3]*x^2+coef(reg5)[2]*x+coef(reg5)[1]
plot(wynagrodzenie, zycie, ylab="Przeciêtna d³ugoœæ ¿ycia",xlab="Wynagrodzenie", type="p", col="blue")
lines(x,linreg5,lwd=2,col="red")

# Model 6: regresja wielomianowa 6 stopnia
reg6=lm(zycie~wynagrodzenie+I(wynagrodzenie^2)+I(wynagrodzenie^3)+I(wynagrodzenie^4)+I(wynagrodzenie^5)+I(wynagrodzenie^6))
summary(reg6)
# R2 = 0.9637
coef(reg6)
# Dostajemy krzyw¹ y=ax^6+bx^5+cx^4+dx^3+ex^2+fx+g
linreg6=coef(reg6)[7]*x^6+coef(reg6)[6]*x^5+coef(reg6)[5]*x^4+coef(reg6)[4]*x^3+coef(reg6)[3]*x^2+coef(reg6)[2]*x+coef(reg6)[1]
plot(wynagrodzenie, zycie, ylab="Przeciêtna d³ugoœæ ¿ycia",xlab="Wynagrodzenie", type="p", col="blue")
lines(x,linreg6,lwd=2,col="red")

# Model 7: regresja wielomianowa 7 stopnia
reg7=lm(zycie~wynagrodzenie+I(wynagrodzenie^2)+I(wynagrodzenie^3)+I(wynagrodzenie^4)+I(wynagrodzenie^5)+I(wynagrodzenie^6)+I(wynagrodzenie^7))
summary(reg7)
# R2 = 0.9642
coef(reg7)
# Dostajemy krzyw¹ y=ax^7+bx^6+cx^5+dx^4+ex^3+fx^2+gx+h
linreg7=coef(reg7)[8]*x^7+coef(reg7)[7]*x^6+coef(reg7)[6]*x^5+coef(reg7)[5]*x^4+coef(reg7)[4]*x^3+coef(reg7)[3]*x^2+coef(reg7)[2]*x+coef(reg7)[1]
plot(wynagrodzenie, zycie, ylab="Przeciêtna d³ugoœæ ¿ycia",xlab="Wynagrodzenie", type="p", col="blue")
lines(x,linreg7,lwd=2,col="red")

# Model 8: regresja wielomianowa 8 stopnia
reg8=lm(zycie~wynagrodzenie+I(wynagrodzenie^2)+I(wynagrodzenie^3)+I(wynagrodzenie^4)+I(wynagrodzenie^5)+I(wynagrodzenie^6)+I(wynagrodzenie^7)+I(wynagrodzenie^8))
summary(reg8)
# R2 = 0.9642
coef(reg8)
# Dostajemy krzyw¹ y=ax^6+bx^5+cx^4+dx^3+ex^2+fx+g
linreg8=coef(reg8)[9]*x^8+coef(reg8)[8]*x^7+coef(reg8)[7]*x^6+coef(reg8)[6]*x^5+coef(reg8)[5]*x^4+coef(reg8)[4]*x^3+coef(reg8)[3]*x^2+coef(reg8)[2]*x+coef(reg8)[1]
plot(wynagrodzenie, zycie, ylab="Przeciêtna d³ugoœæ ¿ycia",xlab="Wynagrodzenie", type="p", col="blue")
lines(x,linreg8,lwd=2,col="red")
data<-data.frame(X=wynagrodzenie,Y=zycie)
rmse <- function(error)
{
  sqrt(mean(error^2))
}
predictedY<-predict(reg8,data$X)
errorlm<-data$Y-predictedY
lm8RMSE <- rmse(errorlm) 
lm8RMSE



# Model 9: svm
library(e1071)

data
model <- svm(Y ~ X , data, kernel="radial", epsilon=0.11, cost=5 )

predictedY_svr <- predict(model, data)
plot(wynagrodzenie, zycie, ylab="Przeciêtna d³ugoœæ ¿ycia",xlab="Wynagrodzenie", type="p", col="blue")
points(data$X, predictedY, col = "red", pch=4)

rmse <- function(error)
{
  sqrt(mean(error^2))
}
errorsvr <- data$Y - predictedY_svr
svrPredictionRMSE <- rmse(errorsvr) 
svrPredictionRMSE
svm.accuracy(predictedY_svr,data$Y)


# Model 9: Regresja wyk³adnicza
regw=lm(log(zycie)~wynagrodzenie)
summary(regw)
coef(regw)
b0=exp(coef(regw)[1])
b1=exp(coef(regw)[2])
nielin=nls(zycie~a0*a1^wynagrodzenie,start=list(a0=b0,a1=b1))
summary(nielin)
coef(nielin)
x=seq(min(wynagrodzenie),max(wynagrodzenie),length=1000)
regwyk=coef(nielin)[1]*coef(nielin)[2]^x
plot(wynagrodzenie,zycie, ylab="Przeciêtna d³ugoœæ ¿ycia",xlab="Wynagrodzenie",col="blue")
lines(x,regwyk,lwd=3,col="red")

# Sprawdzamy, który z siedmiu przedstawionych wy¿ej modeli regresji jest najlepszy,
# poprzez porównanie wspó³czynnika AIC (Akaike’s Information Criterion). Im mniejszy 
# wspó³czynnik, tym dopasowanie lepsze.
AIC(reglin,regkw,regsz,reg4,reg5,reg6,nielin)
# w naszym przypadku najlepszy wynik uzyskaliœmy dla regresji wielomianowej 3 stopnia


#--------------------------------Tworzenie szeregow czasowych-----------------------------
is.ts(zycie)
is.ts(wynagrodzenie)

zycie1 <- ts(zycie,start=c(2008,1),frequency=4)
wynagrodzenie1 <- ts(wynagrodzenie,start=c(2008,1),frequency=4)
ts.plot(zycie1, main="Przeciêtna d³ugoœæ ¿ycia w Polsce",xlab="Czas",ylab="D³ugoœæ ¿ycia")
ts.plot(wynagrodzenie1,main="Wynagrodzenie w Polsce",xlab="Czas", ylab="Wynagrodzenie")

decom_wynagrodzenie=decompose(wynagrodzenie1)
plot(decom_wynagrodzenie,col="blue")

decom_zycie=decompose(zycie1)
plot(decom_zycie,col="blue")

#----------------------------------ANOVA-------------------------------------------------

#dla dlugosci zycia

a=ts(zycie,freq=4,start=c(2008,1))
is.ts(a)
T=rep(c(2008:2017),rep(4,10))
length(T)
length(a)
T=factor(T)
S=rep(1:4,10)
S=factor(S)
anova(lm(a~T+S))
####!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Na podstawie wyników mo¿emy stwierdziæ (p-value < 2.2e-16 )
# W celu sprawdzenia wystêpowania sezonowoœci poprzez zró¿nicowanie wyeliminujemy tendencjê rozwojow¹.

anova(lm(diff(a)~T[-1]+S[-1]))

# Trend zosta³ wyeliminowany. Poniewa¿ p-value przy S jest bardzo ma³a, stwierdzamy, ¿e w naszym szeregu wystêpuje sezonowoœæ.
#####!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# dla wynagrodzenia
b=ts(wynagrodzenie,freq=4,start=c(2008,1))
is.ts(b)
T=rep(c(2008:2017),rep(4,10))
length(T)
length(b)
T=factor(T)
S=rep(1:4,10)
S=factor(S)
anova(lm(b~T+S))
####!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Na podstawie wyników mo¿emy stwierdziæ (p-value < 2.2e-16 )
# W celu sprawdzenia wystêpowania sezonowoœci poprzez zró¿nicowanie wyeliminujemy tendencjê rozwojow¹.

anova(lm(diff(b)~T[-1]+S[-1]))

# Trend zosta³ wyeliminowany. Poniewa¿ p-value przy S jest bardzo ma³a, stwierdzamy, ¿e w naszym szeregu wystêpuje sezonowoœæ.
#####!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



#-----------------------------------------PROGNOZOWANIE------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
#Zycie

#-------------------------------PROGNOZOWANIE METOD¥ ETS-------------------------------------------------------------

etst= ets(a,model="ZZZ",damped=NULL) # Utworzenie modelu
etst
prognoza_etst=forecast(etst,h=frequency(a))
prognoza_etst
plot(prognoza_etst,xlab="Czas",ylab="Zycie")


#---------------------------------------------ARIMA-----------------------------------------------------------

a.a=auto.arima(a)
summary(a.a)
plot(forecast(a.a,h=12), col="black",lwd=1,xlab="Czas",ylab="Zycie")

#----------------------------------------SPRAWDZANIE SKUTECZNOŒCI PREDYKCJI------------------------------------
# która metoda - predykcja jest lepsza

accuracy(etst)
accuracy(a.a)

AIC(etst,a.a)

#Jeszcze jedn¹ mo¿liwoœci¹ porównania modeli jest metoda testów statystycznych.
#Przetestujemy hipotezy dotycz¹ce lepszej trafnoœci modelu ARIMA za pomoc¹ testu
#Diebolda-Mariano. Hipotez¹ zerow¹ bêdzie taka sama dok³adnoœæ obu modeli, a
#hipotez¹ alternatywn¹ bêdzie wiêksza dok³adnoœæ modelu ARIMA, na co wskazuje 
#porównanie wartoœci AIC.

dm.test(resid(etst),resid(a.a), alternative="greater", h=12)
#Nie ma podstaw do odrzucenia hipotezy zerowej, zatem wynik nie jest rozstrzygaj¹cy
#na korzyœæ ¿adnego modelu.





#Wynagrodzenie

#-------------------------------PROGNOZOWANIE METOD¥ ETS-------------------------------------------------------------

etst_b= ets(b,model="ZZZ",damped=NULL) # Utworzenie modelu
etst_b
prognoza_etst_b=forecast(etst_b,h=frequency(b))
prognoza_etst_b
plot(prognoza_etst_b,xlab="Czas",ylab="Wynagrodzenie")

#---------------------------------------------ARIMA-----------------------------------------------------------

b.b=auto.arima(b)
summary(b.b)
plot(forecast(b.b,h=12), col="black",lwd=1,xlab="Czas",ylab="Wynagrodzenie")

#----------------------------------------SPRAWDZANIE SKUTECZNOŒCI PREDYKCJI------------------------------------
# która metoda - predykcja jest lepsza

accuracy(etst_b)
accuracy(b.b)

AIC(etst_b,b.b)

#Jeszcze jedn¹ mo¿liwoœci¹ porównania modeli jest metoda testów statystycznych.
#Przetestujemy hipotezy dotycz¹ce lepszej trafnoœci modelu ARIMA za pomoc¹ testu
#Diebolda-Mariano. Hipotez¹ zerow¹ bêdzie taka sama dok³adnoœæ obu modeli, a
#hipotez¹ alternatywn¹ bêdzie wiêksza dok³adnoœæ modelu ARIMA, na co wskazuje 
#porównanie wartoœci AIC.

dm.test(resid(etst_b),resid(b.b), alternative="greater", h=12)
#Nie ma podstaw do odrzucenia hipotezy zerowej, zatem wynik nie jest rozstrzygaj¹cy
#na korzyœæ ¿adnego modelu.
