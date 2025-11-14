#learing rate
r = 0.1

#Input
a4=1.5

#Random Values
w0 = 0.1
w1 = 0.1
w2 = 0.1
w3 = 0.1

a0 = 0.1
a1 = 0.1
a2 = 0.1
a3 = 0.1

#Zielwert y = 0.5

y=0.5

for x in range (0,400):
    
    #Vorwaertspfad
    a0 = a1*w0   
    a1 = w1*a2
    a2 = w2 * a3
    a3 = w3*a4
    
    #Berechnung der Gewichte
    w0=w0-r*a1*2*(a0-y) 
    
    w1=w1-r * a2*w0*2*(a0-y)
    
    w2=w2 -r * a3 *w1*w0* 2*(a0-y)
    
    w3=w3 -r * a4 *w2*w1*w0* 2*(a0-y)
   
    print ("W0 = ",f"{w0:7.4f}"," W1 = ",f"{w1:7.4f}", " W2 = ",f"{w1:7.4f}"," W3 = ",f"{w1:7.4f}", "   Fehler = ",f"{(a0-y):7.4f}")
 
print("Check", a4*w3*w2*w1*w0) 
