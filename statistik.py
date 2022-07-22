# phyton3
# Letzte Änderung: 2021-02-12
# J.H

''' 
Klassen:
1. Statistik  - Zur Auswertung der Vorhersgeergebnisse
2. Funktionen - Zur Berechnung von Funktionswerten
3. Ausgleich  - Punkte der Ausgleichsfunktion
VERWENDUNG: import MODULE.JH.statistik as STAT   
-----------------------------------------
'''

import numpy as np
from scipy.stats import linregress
import scipy as sp
import scipy as stats
import torch ## for pytorch
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class Statistik():
    ''' 1. Auswertung von Vorhersageergebnissen
    Parameter:
    bezeichnung - Bezeichnung des merkmals, Attributs, Metadatums
        verbose - True/False Schalter für Kontrollausgaben '''
    def __init__(self, bezeichnung='', verbose=False):
        ''' KONSTRUKTOR mit Attributen '''
        super(Statistik, self).__init__()
        self.bezeichnung=bezeichnung
        self.verbose=verbose  # um ggf. etwas auszugeben
    
    # region METHODEN ####

    def confidence_interval (self, data, confidence=0.95):
        '''Ermittelt den Vertrauensbereich (confidence interval) der Datenmenge unter Berücksichtigung der
        Anzahl der Testwerte (t-Verteilung).
        Parameter:
            data       - Datenfeld, eindimensional
            confidence - Faktor der Wahrscheinlichkeit, das der Wert in dem Vertrauensbereich liegt
        Return: 
            mean - Mittelwert
            std  - Standardabweichung der Datenmenge
            confidenze - Unsicherheit +- als Betrag
            confidenceAbove - obere Grenze
            confidenceBelow - untere Grenze des Vertauensbereichs    '''
        #data = 1.0*np.array(data)
        n = len(data)
        std = np.std(data)
        # t-Verteilung zur Berücksichtigung der Testanzahl
        confidence = std * sp.stats.t._ppf((1+confidence)/2., n-1)
        mean=np.mean(data)
        confidenceAbove= mean + confidence
        confidenceBelow= mean - confidence
        return mean, std, confidence, confidenceAbove, confidenceBelow

    def vertrauensbereiche(self, labels, progDifs, confidence, areasAnzahl=11):
        ''' Ermittelt die Vertrauensbereiche abschnittsweise einer Datenmenge.
        Parameter:
            labesl   - Basiswerte, Vorgabewerte, alle Daten
            progDifs - Differenz der Prognosewerte zum Basiswert bzw. zum Vorgabewert
            confidence - Faktor der Wahrscheinlichkeit, das der Wert in dem Vertrauensbereich liegt
            areaAnzahl - Anzahl der Abschnitte (Teilbereiche), Aufteilung des Definitionsbereichs in Teilbereich
        Return: 
            LabelMeans      - Bereichsmittelwerte der Vorgabewerte (Feld)
            confidencesAbove - Obere Vertrauensgrenze für progDifs (Feld) 
            confidencesBelow - Untere Vertrauensgrenze für progDifs (Feld) '''
        print('------ Vertrauensbereich --------')
        # Indexsortierung, liefert die Indizes in der für eine sortierte Liste, ein sortiertes Array.      
        indexes=np.argsort(labels) 
        if self.verbose:
            print(' {:>5} | {:>12} |{:>12} |'.format("Index","Label","Prognoseabweichung"))
            for i in indexes: print(' {:5d} | {:12.4f} | {:11.4f} |'.format(i, labels[i], progDifs[i]))
        wertepaareAnzahl=len(indexes)
        areasGroesse=wertepaareAnzahl // areasAnzahl
        wertepaareRestAnzahl= wertepaareAnzahl % areasAnzahl
        if self.verbose: 
            print('Anzahl der Wertepaare: ',wertepaareAnzahl)
            print('Klassengröße: ', areasGroesse)
            print('Wertepaare-Restanzahl: ',wertepaareRestAnzahl, ' bleibt zunächst unberücksichtigt')
        # Feld für Klassenaufteilung vorbereiten
        areas= np.empty(((areasAnzahl,areasGroesse,2)))
        print('shape', areas.shape)
        # den Klassen zuordnen
        k=0
        j=0
        for i in indexes:
            if k >= areasAnzahl: break # Rest bleibt zunächst unberücksichtigt   
            areas[k,j,0]=labels[i]
            areas[k,j,1]=progDifs[i]
            j +=1
            if j>=areasGroesse:
                j=0
                k+=1
        #print(areas)          
        if self.verbose:
            for k in range(areasAnzahl):
                print(' {:3s} | {:3s} | {:10s} | {:10s} |'.format('k','j', 'Label','Prognoseabweichung'))    
                for j in range(areasGroesse):
                    print(' {:3d}| {:3d} | {:10.3f} | {:10.3f} ||'.format(k, j, areas[k,j,0],areas[k,j,1]))
        # Bereichseise Ermittlung der Konfidenz             
        LabelMeans=np.empty(areasAnzahl)
        Means= np.empty(areasAnzahl)
        Stds =np.empty(areasAnzahl)
        confidences=np.empty(areasAnzahl)
        confidencesAbove=np.empty(areasAnzahl)
        confidencesBelow=np.empty(areasAnzahl)
        for k in range(areasAnzahl):
            LabelMeans[k]=np.mean(areas[k,:,0]) # für die Abszisse
            Means[k], Stds[k], confidences[k], confidencesAbove[k], confidencesBelow[k] =self.confidence_interval(areas[k,:,1], confidence)
        if self.verbose:
            print('LabelMeans ',LabelMeans)
            print('Mean  ',Means)
            print('Std   ', Stds)
            print('Conf  ',confidences) 
            print('Above  ',confidencesAbove) 
            print('Below ',confidencesBelow) 
        return LabelMeans, confidencesAbove, confidencesBelow
    
    def lin_reg(self, x, y):
        ''' Regression für die Vertrauensbereichgrenzen
        Die Testdaten wurden in mehrere Bereich aufgeteilt, zum Beispiel in 11 Bereiche. Für jeden Bereich wurde der
        Vertrauensintervall separat ermittelt, so das für jeden Bereich jeweils ein Punkt für den oberen und für den 
        unteren Grenzwert vorliegt. Eine Regressionsanalyse dient zum Darstellen der Vertrauensgrenzen 
        für alle Daten. Dadurch kann die Größe des Vertrauensbereich zum Beispiel in Abhängigkeit von der Entfernung
        eines Objektes von der Kamera dargestellt werden.
        ------------------------------------------------------------------------------------------------------------
        Lineare Regression
        https://realpython.com/linear-regression-in-python/
        y = b0 + b1 x
        Parameter:
            x - Feld mit Werten
            y - Feld mit Werten
        Return:
            b0, model.intercept_[0] - Koeffizent der Ausgleichsgeraden
            b1, model.coef_[0,0]    - Koeffizent der Ausgleichsgeraden '''
        x=np.array(x).reshape((-1,1))
        y=np.array(y).reshape((-1,1))
        model=LinearRegression()
        model.fit(x,y)
        model = LinearRegression().fit(x, y)
        r_sq = model.score(x, y)
        if self.verbose:
            print('coefficient of determination (Bestimmtheitsmaß) R²:', r_sq)
            print('intercept b0:', model.intercept_)
            print('slope     b1:', model.coef_)
        return [model.intercept_[0], model.coef_[0,0]]

    
    def square_reg(self, x, y):
        ''' Polynomregression mit Scikit-Lernen zweiten Grades
        https://realpython.com/linear-regression-in-python/         
        y = b0 + b1 x + b2 x²
        Parameter:
            x - Feld mit Werten
            y - Feld mit Werten
        Return:
            b0, model.intercept_[0] - Koeffizent der Ausgleichsfunktion
            b1, model.coef_[0,0]    - Koeffizent der Ausgleichsfunktion 
            b2, model.coef_[0,1]    - Koeffizent der Ausgleichsfunktion'''
        x=np.array(x).reshape((-1,1))
        y=np.array(y).reshape((-1,1))
        transformer = PolynomialFeatures(degree=2, include_bias=False)
        transformer.fit(x)
        x_ = transformer.transform(x)
        model = LinearRegression().fit(x_, y)
        r_sq = model.score(x_, y)
        if self.verbose:
            print('coefficient of determination:', r_sq)
            print('intercept b0:', model.intercept_)
            print('coefficients b1, b2:', model.coef_)
        return [model.intercept_[0], model.coef_[0,0], model.coef_[0,1]]
    
    def quad_reg(self, x, y):
        ''' Polynomregression mit Scikit-Lernen zweiten Grades
        https://realpython.com/linear-regression-in-python/         
        y = b0 + b1 x + b2 x²
        Parameter:
            x - Feld mit Werten
            y - Feld mit Werten
        Return:
            b0, model.intercept_[0] - Koeffizent der Ausgleichsfunktion
            b1, model.coef_[0,0]    - Koeffizent der Ausgleichsfunktion 
            b2, model.coef_[0,1]    - Koeffizent der Ausgleichsfunktion'''
        x=np.array(x).reshape((-1,1))
        y=np.array(y).reshape((-1,1))
        transformer = PolynomialFeatures(degree=4, include_bias=False)
        transformer.fit(x)
        x_ = transformer.transform(x)
        model = LinearRegression().fit(x_, y)
        r_sq = model.score(x_, y)
        if self.verbose:
            print('coefficient of determination:', r_sq)
            print('intercept b0:', model.intercept_)
            print('coefficients b1, b2:', model.coef_)
        return [model.intercept_[0], model.coef_[0,0], model.coef_[0,1], model.coef_[0,2], model.coef_[0,3]]
    
    
    def class_counter(self, listDelta ):
        ''' Kurze Statistik
        Es folgt eine kurze Auswertung der Daten. Eine Ausführliche Darstellung der Ergebnisse folgt in einem gesonderten Programm (4.).
        Die Auflösung der Orientierung erfolgt zunächst mit einer Genauigkeit von einem Grad.
        Eine Reduzierung der Auflösung ist durch Zusammenfassen mehrere Klassen, zum Beispiel für eine Auflösung von 5 Grad, möglich.
        -> Liefert Haufigkeit in den Intervallen 0, +-5, +-10
        Parameter:
            listDelta - List der Differenzen zwischen Labels und Prognose
        Return:
            countEqual      - Anzahl der 0-Differenzen
            countEqual_u5   - Anzahl der Werte im Vertrauensbereich +-5
            countEqual_u10  - Anzahl der Werte im Vertrauensbereich +-10 '''
        countEqual=0
        countEqual_u5=0
        countEqual_u10=0
        for delta in listDelta:
            if delta == 0: countEqual +=1 # zur Kontrolle
            if abs(delta) <=5:   countEqual_u5+=1
            if abs(delta) <=10: countEqual_u10+=1
        return countEqual, countEqual_u5, countEqual_u10


    def short_verteilung(self, dirSave,list_D_labels):
        ''' Ermittelt eine einfache Verteilung bezüglich der Abweichung zwischen Label und Prognose.
        Eine Ausführliche Darstellung der Ergebnisse folgt in einem gesonderten Programm.
        Die Auflösung der Orientierung erfolgt zunächst mit einer Genauigkeit von einem Grad.
        Eine Reduzierung der Auflösung ist durch Zusammenfassen mehrere Klassen möglich.
        Parameter:
            dirSave       - Verzeichnis in dem das Ergebnis abgeschpeichert wird
            list_D-Labels - Liste der Differenzen zwischen Label und Prognose '''
        countAll=len(list_D_labels)
        countEqual, countEqual_u5, countEqual_u10= self.class_counter(list_D_labels)
        # Ausgabewünsche zusammenstellen und formatieren
        line1="--"
        line2="--"
        line3="--"
        if countAll>0:
            line1=('Anzahl der Proben {}, davon korrekt {}, Verhältnis: {:6.4f}\n'.format(countAll,countEqual,countEqual/countAll))
            line2=('Anzahl der Proben {}, davon mit Unsicherheit +-5: {}, Verhältnis: {:6.4f}\n'.format(countAll,countEqual_u5,countEqual_u5/countAll))
            line3=('Anzahl der Proben {}, davon mit Unsicherheit +-10: {}, Verhältnis: {:6.4f}\n'.format(countAll,countEqual_u10,countEqual_u10/countAll))
        if self.verbose:
            # Ausgeben
            print()
            print(line1)
            print(line2)
            print(line3)
        # Textdatei speichern/erweitern
        fileName=dirSave+self.bezeichnung+"_Statistik_"+'.txt'
        statistik=open(fileName,"w")
        statistik.write(line1)
        statistik.write(line2)
        statistik.write(line3)
        statistik.close()


    def calculate_img_stats_full(self, dataset):
        ''' -> noch nicht getestet
        Ermittelt Mittelwert und Standardabweichung über ein gesamtes Datenset
        Parameter:
            dataset - Datenset
        Return
            img_mean - Mittelerer Intensitätswert (r,g,b)
            img_std  - Standardabeichung der Intensität (r,g,b)  '''
        imgs_ = torch.stack([img for img,_ in dataset],dim=3)
        imgs_ = imgs_.view(3,-1)
        imgs_mean = imgs_.mean(dim=1)
        imgs_std = imgs_.std(dim=1)
        return imgs_mean,imgs_std	
#.........................................................................


    def confidence_funktion(self, labels, diffs, confidence, areasAnzahl=13, fktTyp='sqr'):
        ''' Zusammenstellung mehrer Funktionen zur Berechnung der oberen und unteren Vertrauensbereichsfunktion
        Parameter:
            labels - Targets, Definitionsbereich
            diffs  - Prognoseabweichungen , Wertebereich
            confidence -  Wahrscheinlichkeit, das der Wert in dem Vertrauensbereich liegt (95%)
            areasAnzahl- Aufteilung des Definitionsbereiches in Teilbereiche. Für jeden
                            Teilbereich wird der Vertauensintervall unabhängig berechnet.
            fktTyp     - lin - Linear, sqr - Quadratisch
                         Die Stützpunkte der Teilbereiche für den oberen und den unteren 
                         Vertrauensbereichsgrenze werdn zur berechnung einer regressionsfunktion verwendet.
        Return:
            x - Definitionswerte, Labels (nicht verwechseln mit dem np.array)
            yAbove - Koeffizienten der oberen Vertrauensbereichsfunktion
            yBelow - Koeffizienten der unteren Vertrauensbereichsfunktion 
        '''
        # Felder mit Stützpunkten der einzelnen Bereiche
        labelsMeans, confidencesAbove, confidencesBelow=\
            self.vertrauensbereiche(labels=labels, progDifs=diffs, confidence=confidence, areasAnzahl=areasAnzahl)
        # Mit den Funktionsparametern werden jetzt Punkte der Ausgleichsfunktionen berechnet
        x=labelsMeans.tolist()  # x-Werte aus Ausgleichsfunktionen der Stützpunkte
        x.append(max(labels))  # Erweiterung um einen Stützpunkt
        x.insert(0,min(labels)) # noch eine Erweiterung

        # Funktionsparameter für oberen und unteren Funktionsverlauf
        fkt=Funktionen(self.verbose)
        if fktTyp=='sqr':
            # print('obere Vertrauensgrenze180: {},\n untere Vertrauensgrenze180: {}'.format(location.Above180, location.Below180))
            # print('---- square Reg: y = b0, b1*x + b2*x² ------------------')
            sqrAbove = self.square_reg(x=labelsMeans, y=confidencesAbove)
            sqrBelow = self.square_reg(x=labelsMeans, y=confidencesBelow)
            yAbove=fkt.squareValues(sqrAbove, x)
            yBelow=fkt.squareValues(sqrBelow, x)
        else:
            # print('Location: ', location.name)
            # print('---- lin. Reg: y = b0 + b1 *x -------------------')
            linAbove= self.lin_reg(x=labelsMeans, y=confidencesAbove)
            linBelow= self.lin_reg(x=labelsMeans, y=confidencesBelow)
            yAbove=fkt.linearValues(linAbove, x)
            yBelow=fkt.linearValues(linBelow, x)
        return x, yAbove, yBelow

############################################################################        
#### KLASSE #################################################################

class Funktionen():
    ''' 2. Berechnung von Funktionswerten
    Parameter:
        verbose - True/False Schalter für Kontrollausgaben   '''
    # KONSTRUKTOR mit Attributen
    def __init__(self, verbose=False):
        super(Funktionen, self).__init__()
        self.verbose=verbose  # um ggf. etwas auszugeben

    #region Funktionswerte
    def squareValues(self, B, X):
        ''' Berechnet Funktionswerte einer quadratischen Funktion
        Parameter:
            B (Feld) - B[0], B[1], B[2] -  der quadratischen Funktion
            X (Feld) - X[0], X[1], ..., X[n-1] - Werte
        Return:
            Y (Feld) - Y[0], Y[1], ..., Y[n-1] - Funktionswerte'''
        Y= np.empty(len(X))
        for i in range(len(X)):
            Y[i]=B[0] + B[1] * X[i] + B[2] * X[i] *X[i]
        return Y     

    def linearValues(self, B, X):
        ''' Berechnet Funktionswerte einer lineareb Funktion
        Parameter:
            B (Feld) - B[0], B[1] - Faktoren
            X (Feld) - X[0], X[1], ..., X[n-1] - Werte
        Return:
            Y (Feld) - Y[0], Y[1], ..., Y[n-1] - Funktionswerte'''
        Y= np.empty(len(X))
        for i in range(len(X)):
            Y[i]=B[0] + B[1] * X[i] 
        return Y  
    #endregion
    
############################################################################        
#### KLASSE #################################################################

class Ausgleich():
    ''' 3. Liefert Ausgleichspunkte '''
   # KONSTRUKTOR mit Attributen
    def __init__(self, verbose=False):
        super(Ausgleich, self).__init__()
        self.verbose=verbose  # um ggf. etwas auszugeben
        
    
    def quadratisch(self, x, y, k=[0]):
        '''Ausgleichspunkte mittels qauadratischer Regressions-Funktion und Korrekturmöglichkeit'''
        statist = Statistik()
        y1 = np.zeros(len(y))
        for i in range(len(y)):
            try:
                y1[i] = y[i] + k[i]
            except:
                y1[i] = y[i]
                
                
                
        b0, b1, b2 =statist.square_reg(x, y1)
        n = 100
        X = np.zeros(n)
        Y = np.zeros(n)
        dx = (x[len(x) - 1] - x[0]) / (n - 1)
        X[0] = x[0]
        for i in range(n):
            X[i] = X[0] + dx * i
            Y[i] = b0 + b1 * X[i] + b2 * X[i] * X[i]
        return X, Y

  


        




    


    
