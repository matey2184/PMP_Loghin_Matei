Explicatie Lab9(b si d)
Scopul e sa aflam cam cati clienti in total, n, ar trebui sa avem, stiind cati cumparatori, Y, am vazut si care e probabilitatea lor de cumparare, theta.
Cum am pus bazele modelului:Priorul (Parerea de start despre clienti, n): Ne bazam pe o distributie Poisson(10). Deci, din start, credem ca avem in jur de 10 clienti.
Verosimilitatea (Cumparatorii, Y): Asta e o distributie Binomiala(n, theta). Adica, daca avem n clienti in total, fiecare cumpara cu probabilitatea theta, deci vedem Y cumparatori.
b) Ce fac Y si theta cu distributia pentru n (clientii totali)Practic, distributia P(n | Y, theta) e un compromis: pe de o parte, trage spre prior (n=10), pe de alta parte, trebuie sa fie compatibila cu ce am vazut (Y).
Efectul lui Y (Cumparatorii Observati):Y e o limita minima clara: n (clientii totali) nu poate fi mai mic decat Y (cumparatorii).
Y mare (ex: 10): Daca am vazut 10 cumparatori, n trebuie sa fie cel putin 10. Asta impinge distributia n mai sus, mai ales daca theta e mic 
Y mic (ex: 0): Daca nu am vazut pe nimeni cumparand, n poate sa ramana pe la 10 (media de start), poate chiar putin sub, pentru ca 0 cumparatori sugereaza ca nu e un numar mare de clienti
Efectul lui theta (Probabilitatea de Achizitie):Theta ne zice cat de multi n ne trebuie ca sa producem Y cumparatori.
Theta mic (0.2): Daca probabilitatea de cumparare e mica (doar 2 din 10 cumpara), ca sa vedemi 5 sau 10 cumparatori (Y), ne trebuie foarte multi clienti in total (n).Distributia n se duce mult mai sus decat 10.Devine mai larga (mai multa incertitudine), pentru ca n-ul e estimat departe de parerea noastra initiala (10)
Theta mare (0.5): Daca probabilitatea de cumparare e mare (5 din 10 cumpara), trebuie mai putini clienti (n) ca sa vezi acelasi numar Y.Distributia n ramane mai aproape de 10 (in special pentru Y=5, unde 5/0.5 e exact 10).E mai ingusta (mai putina incertitudine), pentru ca datele (Y) sunt foarte clare in legatura cu n
d) Care e diferenta dintre Posteriorul n si Posteriorul Predictiv Y:Aici e vorba de doua tipuri diferite de necunoscut:| Distributie | $P(n | Y, theta) (Posterior n) | $P(Y* | Y, theta) (Posterior Predictiv Y*) || :--- | :--- | :--- || Ce masoara | Un Parametru (n, numarul total REAL de clienti). | O Observatie Viitoare (Y*, cati cumparatori o sa avem data viitoare)
|| Scop | Inferenta: Aflam cat ar trebui sa fie cauza reala (n). | Predictie: Prezicem un rezultat viitor Y. 
