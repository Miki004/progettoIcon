:- discontiguous sedentario/2, vita_sociale/2, tempo_social/2, pasti/2, beve/2, consumo_calorico/2, fumare/2, metabolismo_alto/2, stress_alto/2, dorme_poco/2, eta_protetta/2, rischio_obesita/1, malattia/2.

% ========= Definizione dello stile di vita =========
vita_sedentaria(P) :- sedentario(P, si).
sportivo(P) :- sedentario(P, no).
vita_sedentaria(P) :- vita_sociale(P, no), tempo_social(P, alto).

% Lo stile di vita a rischio dipende dalla sedentarietà e dal fumo
stile_vita_a_rischio(P) :- vita_sedentaria(P), fumare(P, si).
stile_vita_a_rischio(P) :- vita_sociale(P, no), tempo_social(P, alto).
stile_vita_a_rischio(P) :- dorme_poco(P).

% Una persona ha un metabolismo alto se è giovane
metabolismo_alto(P) :- eta_protetta(P,si).

% Una persona ha un metabolismo alto se è sportiva
metabolismo_alto(P) :- sportivo(P).

% Una persona ha un metabolismo alto se ha una dieta sana e beve molta acqua
metabolismo_alto(P) :- mangia_bene(P), beve_molto(P).

% Una persona ha un metabolismo alto se dorme molto e non è stressata
metabolismo_alto(P) :- dorme_molto(P).

% ========= Definizione della dieta =========
mangia_bene(P) :- consumo_calorico(P, basso).
mangia_male(P) :- consumo_calorico(P, alto).

beve_poco(P) :- beve(P, poco).
beve_molto(P) :- beve(P, molto).

dorme_molto(P) :-dorme(P,molto).
dorme_poco(P) :-dorme(P,poco).

% Una dieta scorretta dipende dalla sedentarietà e dal numero di pasti
dieta_scorretta(P) :- mangia_male(P), beve_poco(P).
dieta_scorretta(P) :- pasti(P, troppi), vita_sedentaria(P).

% Se una persona ha metabolismo alto e fa attività fisica, compensa la dieta
compensa_dieta(P) :- metabolismo_alto(P), sportivo(P).
compensa_dieta(P) :- metabolismo_alto(P), dieta_scorretta(P), beve_molto(P).

protezione_obesita(P) :- sportivo(P), eta_protetta(P,si),dorme_molto(P).
protezione_obesita(P) :- beve_molto(P), mangia_bene(P).

% ========= Classificazione del rischio di obesità =========
rischio_obesita(P) :- stile_vita_a_rischio(P), dieta_scorretta(P).
rischio_obesita(P) :- dieta_scorretta(P), \+ compensa_dieta(P).
rischio_obesita(P) :- stile_vita_a_rischio(P), \+ protezione_obesita(P).

% ========= Associazione tra obesità e malattie =========
% Diabete: legato a obesità e dieta scorretta
malattia(P, diabete) :- rischio_obesita(P), dieta_scorretta(P).

% Ipertensione: legata a obesità, fumo e sedentarietà
malattia(P, ipertensione) :- rischio_obesita(P), fumare(P, si).

% Malattie cardiovascolari: collegate a obesità, fumo, dieta scorretta e poco esercizio fisico
malattia(P, cardiovascolare) :- rischio_obesita(P), fumare(P, si), dieta_scorretta(P),eta_protetta(P,no).
malattia(P, cardiovascolare) :- vita_sedentaria(P), dieta_scorretta(P),eta_protetta(P,no).

% Apnea notturna: legata a obesità e disturbi del sonno
malattia(P, apnea_notturna) :- rischio_obesita(P), dorme_poco(P).

% Sindrome metabolica: una combinazione di obesità, ipertensione e diabete
malattia(P, sindrome_metabolica) :- malattia(P, diabete), malattia(P, ipertensione), rischio_obesita(P).

% ========= Vincoli di integrità =========
% Una persona non può essere sedentaria e sportiva contemporaneamente
errore(P, 'Contraddizione: sedentario e sportivo insieme') :- sedentario(P, si), sportivo(P).

% Una persona non può mangiare bene e avere un alto consumo calorico
errore(P, 'Contraddizione: mangia bene ma ha consumo calorico alto') :- mangia_bene(P), consumo_calorico(P, alto).

% Una persona non può dormire poco e molto contemporaneamente
errore(P, 'Contraddizione: dorme poco e molto') :- dorme_poco(P), dorme_molto(P).

% Una persona non può avere stile di vita sano e essere a rischio obesità
errore(P, 'Contraddizione: stile di vita sano ma a rischio obesità') :-
    protezione_obesita(P),
    rischio_obesita(P).

% ========= Fatti =========


persona(p547).
eta_protetta(p547,si).
sedentario(p547,si).
vita_sociale(p547,si).
tempo_social(p547, basso).
pasti(p547,pochi).
beve(p547,poco).
dorme(p547,poco).
consumo_calorico(p547,alto).
fumare(p547,si).
persona(p524).
eta_protetta(p524,si).
sedentario(p524,si).
vita_sociale(p524,si).
tempo_social(p524, basso).
pasti(p524,pochi).
beve(p524,poco).
dorme(p524,poco).
consumo_calorico(p524,alto).
fumare(p524,si).
persona(p775).
eta_protetta(p775,si).
sedentario(p775,si).
vita_sociale(p775,si).
tempo_social(p775, basso).
pasti(p775,pochi).
beve(p775,poco).
dorme(p775,poco).
consumo_calorico(p775,alto).
fumare(p775,si).
persona(p980).
eta_protetta(p980,si).
sedentario(p980,si).
vita_sociale(p980,si).
tempo_social(p980, basso).
pasti(p980,pochi).
beve(p980,poco).
dorme(p980,poco).
consumo_calorico(p980,alto).
fumare(p980,si).
persona(p959).
eta_protetta(p959,si).
sedentario(p959,si).
vita_sociale(p959,si).
tempo_social(p959, basso).
pasti(p959,pochi).
beve(p959,poco).
dorme(p959,poco).
consumo_calorico(p959,alto).
fumare(p959,si).
persona(p258).
eta_protetta(p258,si).
sedentario(p258,si).
vita_sociale(p258,si).
tempo_social(p258, basso).
pasti(p258,pochi).
beve(p258,poco).
dorme(p258,poco).
consumo_calorico(p258,alto).
fumare(p258,si).
persona(p265).
eta_protetta(p265,si).
sedentario(p265,si).
vita_sociale(p265,si).
tempo_social(p265, basso).
pasti(p265,pochi).
beve(p265,poco).
dorme(p265,poco).
consumo_calorico(p265,alto).
fumare(p265,si).
persona(p641).
eta_protetta(p641,si).
sedentario(p641,si).
vita_sociale(p641,si).
tempo_social(p641, basso).
pasti(p641,pochi).
beve(p641,poco).
dorme(p641,poco).
consumo_calorico(p641,alto).
fumare(p641,si).
persona(p709).
eta_protetta(p709,si).
sedentario(p709,si).
vita_sociale(p709,si).
tempo_social(p709, basso).
pasti(p709,pochi).
beve(p709,poco).
dorme(p709,poco).
consumo_calorico(p709,alto).
fumare(p709,si).
persona(p511).
eta_protetta(p511,si).
sedentario(p511,si).
vita_sociale(p511,si).
tempo_social(p511, basso).
pasti(p511,pochi).
beve(p511,poco).
dorme(p511,poco).
consumo_calorico(p511,alto).
fumare(p511,si).
persona(p824).
eta_protetta(p824,si).
sedentario(p824,no).
vita_sociale(p824,si).
tempo_social(p824, alto).
pasti(p824,troppi).
beve(p824,molto).
dorme(p824,molto).
consumo_calorico(p824,alto).
fumare(p824,si).
persona(p634).
eta_protetta(p634,si).
sedentario(p634,si).
vita_sociale(p634,si).
tempo_social(p634, basso).
pasti(p634,pochi).
beve(p634,poco).
dorme(p634,poco).
consumo_calorico(p634,alto).
fumare(p634,si).
persona(p279).
eta_protetta(p279,si).
sedentario(p279,si).
vita_sociale(p279,si).
tempo_social(p279, basso).
pasti(p279,pochi).
beve(p279,poco).
dorme(p279,poco).
consumo_calorico(p279,alto).
fumare(p279,si).
persona(p96).
eta_protetta(p96,si).
sedentario(p96,no).
vita_sociale(p96,no).
tempo_social(p96, basso).
pasti(p96,troppi).
beve(p96,molto).
dorme(p96,poco).
consumo_calorico(p96,alto).
fumare(p96,si).
persona(p536).
eta_protetta(p536,si).
sedentario(p536,no).
vita_sociale(p536,no).
tempo_social(p536, basso).
pasti(p536,troppi).
beve(p536,molto).
dorme(p536,poco).
consumo_calorico(p536,alto).
fumare(p536,si).
persona(p789).
eta_protetta(p789,si).
sedentario(p789,no).
vita_sociale(p789,no).
tempo_social(p789, basso).
pasti(p789,troppi).
beve(p789,molto).
dorme(p789,poco).
consumo_calorico(p789,alto).
fumare(p789,si).
persona(p959).
eta_protetta(p959,si).
sedentario(p959,no).
vita_sociale(p959,no).
tempo_social(p959, basso).
pasti(p959,troppi).
beve(p959,molto).
dorme(p959,poco).
consumo_calorico(p959,alto).
fumare(p959,si).
persona(p187).
eta_protetta(p187,si).
sedentario(p187,no).
vita_sociale(p187,no).
tempo_social(p187, basso).
pasti(p187,troppi).
beve(p187,molto).
dorme(p187,poco).
consumo_calorico(p187,alto).
fumare(p187,si).
persona(p60).
eta_protetta(p60,si).
sedentario(p60,no).
vita_sociale(p60,no).
tempo_social(p60, basso).
pasti(p60,troppi).
beve(p60,molto).
dorme(p60,poco).
consumo_calorico(p60,alto).
fumare(p60,si).
persona(p18).
eta_protetta(p18,si).
sedentario(p18,no).
vita_sociale(p18,no).
tempo_social(p18, basso).
pasti(p18,troppi).
beve(p18,molto).
dorme(p18,poco).
consumo_calorico(p18,alto).
fumare(p18,si).
persona(p723).
eta_protetta(p723,si).
sedentario(p723,no).
vita_sociale(p723,no).
tempo_social(p723, basso).
pasti(p723,troppi).
beve(p723,molto).
dorme(p723,poco).
consumo_calorico(p723,alto).
fumare(p723,si).
persona(p953).
eta_protetta(p953,si).
sedentario(p953,no).
vita_sociale(p953,no).
tempo_social(p953, basso).
pasti(p953,troppi).
beve(p953,molto).
dorme(p953,poco).
consumo_calorico(p953,alto).
fumare(p953,si).
persona(p82).
eta_protetta(p82,si).
sedentario(p82,no).
vita_sociale(p82,no).
tempo_social(p82, basso).
pasti(p82,troppi).
beve(p82,molto).
dorme(p82,poco).
consumo_calorico(p82,alto).
fumare(p82,si).
persona(p268).
eta_protetta(p268,si).
sedentario(p268,no).
vita_sociale(p268,no).
tempo_social(p268, basso).
pasti(p268,troppi).
beve(p268,molto).
dorme(p268,poco).
consumo_calorico(p268,alto).
fumare(p268,si).
persona(p706).
eta_protetta(p706,si).
sedentario(p706,no).
vita_sociale(p706,no).
tempo_social(p706, basso).
pasti(p706,troppi).
beve(p706,molto).
dorme(p706,poco).
consumo_calorico(p706,alto).
fumare(p706,si).
persona(p641).
eta_protetta(p641,si).
sedentario(p641,no).
vita_sociale(p641,no).
tempo_social(p641, basso).
pasti(p641,troppi).
beve(p641,molto).
dorme(p641,poco).
consumo_calorico(p641,alto).
fumare(p641,si).
persona(p300).
eta_protetta(p300,si).
sedentario(p300,si).
vita_sociale(p300,si).
tempo_social(p300, basso).
pasti(p300,pochi).
beve(p300,poco).
dorme(p300,poco).
consumo_calorico(p300,alto).
fumare(p300,si).
persona(p136).
eta_protetta(p136,no).
sedentario(p136,no).
vita_sociale(p136,no).
tempo_social(p136, basso).
pasti(p136,troppi).
beve(p136,molto).
dorme(p136,molto).
consumo_calorico(p136,alto).
fumare(p136,si).
persona(p705).
eta_protetta(p705,no).
sedentario(p705,no).
vita_sociale(p705,no).
tempo_social(p705, basso).
pasti(p705,troppi).
beve(p705,molto).
dorme(p705,molto).
consumo_calorico(p705,alto).
fumare(p705,si).
persona(p679).
eta_protetta(p679,no).
sedentario(p679,no).
vita_sociale(p679,no).
tempo_social(p679, basso).
pasti(p679,troppi).
beve(p679,molto).
dorme(p679,molto).
consumo_calorico(p679,alto).
fumare(p679,si).
persona(p128).
eta_protetta(p128,no).
sedentario(p128,no).
vita_sociale(p128,no).
tempo_social(p128, basso).
pasti(p128,troppi).
beve(p128,molto).
dorme(p128,molto).
consumo_calorico(p128,alto).
fumare(p128,si).
persona(p928).
eta_protetta(p928,no).
sedentario(p928,no).
vita_sociale(p928,no).
tempo_social(p928, basso).
pasti(p928,troppi).
beve(p928,molto).
dorme(p928,molto).
consumo_calorico(p928,alto).
fumare(p928,si).
persona(p820).
eta_protetta(p820,si).
sedentario(p820,si).
vita_sociale(p820,si).
tempo_social(p820, basso).
pasti(p820,pochi).
beve(p820,poco).
dorme(p820,poco).
consumo_calorico(p820,alto).
fumare(p820,si).
persona(p81).
eta_protetta(p81,si).
sedentario(p81,si).
vita_sociale(p81,no).
tempo_social(p81, basso).
pasti(p81,troppi).
beve(p81,molto).
dorme(p81,poco).
consumo_calorico(p81,alto).
fumare(p81,si).
persona(p563).
eta_protetta(p563,si).
sedentario(p563,si).
vita_sociale(p563,no).
tempo_social(p563, basso).
pasti(p563,troppi).
beve(p563,molto).
dorme(p563,poco).
consumo_calorico(p563,alto).
fumare(p563,si).
persona(p242).
eta_protetta(p242,si).
sedentario(p242,si).
vita_sociale(p242,no).
tempo_social(p242, basso).
pasti(p242,troppi).
beve(p242,molto).
dorme(p242,poco).
consumo_calorico(p242,alto).
fumare(p242,si).
persona(p323).
eta_protetta(p323,si).
sedentario(p323,si).
vita_sociale(p323,no).
tempo_social(p323, basso).
pasti(p323,troppi).
beve(p323,molto).
dorme(p323,poco).
consumo_calorico(p323,alto).
fumare(p323,si).
persona(p607).
eta_protetta(p607,si).
sedentario(p607,si).
vita_sociale(p607,no).
tempo_social(p607, basso).
pasti(p607,troppi).
beve(p607,molto).
dorme(p607,poco).
consumo_calorico(p607,alto).
fumare(p607,si).
persona(p804).
eta_protetta(p804,si).
sedentario(p804,si).
vita_sociale(p804,no).
tempo_social(p804, basso).
pasti(p804,troppi).
beve(p804,molto).
dorme(p804,poco).
consumo_calorico(p804,alto).
fumare(p804,si).
persona(p963).
eta_protetta(p963,si).
sedentario(p963,no).
vita_sociale(p963,no).
tempo_social(p963, basso).
pasti(p963,troppi).
beve(p963,molto).
dorme(p963,molto).
consumo_calorico(p963,alto).
fumare(p963,si).
persona(p584).
eta_protetta(p584,si).
sedentario(p584,no).
vita_sociale(p584,no).
tempo_social(p584, basso).
pasti(p584,troppi).
beve(p584,molto).
dorme(p584,molto).
consumo_calorico(p584,alto).
fumare(p584,si).
persona(p480).
eta_protetta(p480,si).
sedentario(p480,no).
vita_sociale(p480,no).
tempo_social(p480, basso).
pasti(p480,troppi).
beve(p480,molto).
dorme(p480,molto).
consumo_calorico(p480,alto).
fumare(p480,si).
persona(p370).
eta_protetta(p370,si).
sedentario(p370,no).
vita_sociale(p370,no).
tempo_social(p370, basso).
pasti(p370,troppi).
beve(p370,molto).
dorme(p370,molto).
consumo_calorico(p370,alto).
fumare(p370,si).
persona(p972).
eta_protetta(p972,si).
sedentario(p972,no).
vita_sociale(p972,no).
tempo_social(p972, basso).
pasti(p972,troppi).
beve(p972,molto).
dorme(p972,molto).
consumo_calorico(p972,alto).
fumare(p972,si).
persona(p716).
eta_protetta(p716,si).
sedentario(p716,no).
vita_sociale(p716,no).
tempo_social(p716, basso).
pasti(p716,troppi).
beve(p716,molto).
dorme(p716,molto).
consumo_calorico(p716,alto).
fumare(p716,si).
persona(p253).
eta_protetta(p253,si).
sedentario(p253,no).
vita_sociale(p253,no).
tempo_social(p253, basso).
pasti(p253,troppi).
beve(p253,molto).
dorme(p253,molto).
consumo_calorico(p253,alto).
fumare(p253,si).
persona(p240).
eta_protetta(p240,si).
sedentario(p240,no).
vita_sociale(p240,no).
tempo_social(p240, basso).
pasti(p240,troppi).
beve(p240,molto).
dorme(p240,molto).
consumo_calorico(p240,alto).
fumare(p240,si).
persona(p451).
eta_protetta(p451,si).
sedentario(p451,no).
vita_sociale(p451,no).
tempo_social(p451, basso).
pasti(p451,troppi).
beve(p451,molto).
dorme(p451,molto).
consumo_calorico(p451,alto).
fumare(p451,si).
persona(p599).
eta_protetta(p599,si).
sedentario(p599,no).
vita_sociale(p599,no).
tempo_social(p599, basso).
pasti(p599,troppi).
beve(p599,molto).
dorme(p599,molto).
consumo_calorico(p599,alto).
fumare(p599,si).
persona(p663).
eta_protetta(p663,si).
sedentario(p663,no).
vita_sociale(p663,no).
tempo_social(p663, basso).
pasti(p663,troppi).
beve(p663,molto).
dorme(p663,molto).
consumo_calorico(p663,alto).
fumare(p663,si).
persona(p379).
eta_protetta(p379,si).
sedentario(p379,si).
vita_sociale(p379,si).
tempo_social(p379, basso).
pasti(p379,pochi).
beve(p379,poco).
dorme(p379,poco).
consumo_calorico(p379,alto).
fumare(p379,si).
persona(p772).
eta_protetta(p772,si).
sedentario(p772,si).
vita_sociale(p772,si).
tempo_social(p772, basso).
pasti(p772,pochi).
beve(p772,poco).
dorme(p772,poco).
consumo_calorico(p772,alto).
fumare(p772,si).
persona(p292).
eta_protetta(p292,si).
sedentario(p292,si).
vita_sociale(p292,si).
tempo_social(p292, basso).
pasti(p292,pochi).
beve(p292,poco).
dorme(p292,poco).
consumo_calorico(p292,alto).
fumare(p292,si).
persona(p891).
eta_protetta(p891,si).
sedentario(p891,si).
vita_sociale(p891,si).
tempo_social(p891, basso).
pasti(p891,pochi).
beve(p891,poco).
dorme(p891,poco).
consumo_calorico(p891,alto).
fumare(p891,si).
persona(p922).
eta_protetta(p922,si).
sedentario(p922,si).
vita_sociale(p922,si).
tempo_social(p922,basso).
pasti(p922,pochi).
beve(p922,poco).
dorme(p922,poco).
consumo_calorico(p922,alto).
fumare(p922,si).
persona(p80).
eta_protetta(p80,si).
sedentario(p80,si).
vita_sociale(p80,si).
tempo_social(p80,basso).
pasti(p80,pochi).
beve(p80,poco).
dorme(p80,poco).
consumo_calorico(p80,alto).
fumare(p80,si).
persona(p425).
eta_protetta(p425,si).
sedentario(p425,no).
vita_sociale(p425,si).
tempo_social(p425, basso).
pasti(p425,pochi).
beve(p425,poco).
dorme(p425,molto).
consumo_calorico(p425,alto).
fumare(p425,si).
persona(p675).
eta_protetta(p675,si).
sedentario(p675,no).
vita_sociale(p675,si).
tempo_social(p675, basso).
pasti(p675,pochi).
beve(p675,poco).
dorme(p675,molto).
consumo_calorico(p675,alto).
fumare(p675,si).
persona(p637).
eta_protetta(p637,si).
sedentario(p637,no).
vita_sociale(p637,si).
tempo_social(p637, basso).
pasti(p637,pochi).
beve(p637,poco).
dorme(p637,molto).
consumo_calorico(p637,alto).
fumare(p637,si).
persona(p797).
eta_protetta(p797,si).
sedentario(p797,si).
vita_sociale(p797,si).
tempo_social(p797, basso).
pasti(p797,pochi).
beve(p797,poco).
dorme(p797,poco).
consumo_calorico(p797,alto).
fumare(p797,si).
persona(p13).
eta_protetta(p13,si).
sedentario(p13,si).
vita_sociale(p13,si).
tempo_social(p13, basso).
pasti(p13,pochi).
beve(p13,poco).
dorme(p13,molto).
consumo_calorico(p13,alto).
fumare(p13,si).
persona(p266).
eta_protetta(p266,si).
sedentario(p266,si).
vita_sociale(p266,si).
tempo_social(p266, basso).
pasti(p266,troppi).
beve(p266,poco).
dorme(p266,molto).
consumo_calorico(p266,alto).
fumare(p266,si).
persona(p368).
eta_protetta(p368,si).
sedentario(p368,si).
vita_sociale(p368,si).
tempo_social(p368, basso).
pasti(p368,troppi).
beve(p368,poco).
dorme(p368,molto).
consumo_calorico(p368,alto).
fumare(p368,si).
persona(p87).
eta_protetta(p87,si).
sedentario(p87,si).
vita_sociale(p87,si).
tempo_social(p87, basso).
pasti(p87,troppi).
beve(p87,poco).
dorme(p87,molto).
consumo_calorico(p87,alto).
fumare(p87,si).
persona(p203).
eta_protetta(p203,si).
sedentario(p203,si).
vita_sociale(p203,si).
tempo_social(p203, basso).
pasti(p203,troppi).
beve(p203,poco).
dorme(p203,molto).
consumo_calorico(p203,alto).
fumare(p203,si).
persona(p844).
eta_protetta(p844,si).
sedentario(p844,si).
vita_sociale(p844,si).
tempo_social(p844, basso).
pasti(p844,troppi).
beve(p844,poco).
dorme(p844,molto).
consumo_calorico(p844,alto).
fumare(p844,si).
persona(p937).
eta_protetta(p937,si).
sedentario(p937,no).
vita_sociale(p937,no).
tempo_social(p937, basso).
pasti(p937,troppi).
beve(p937,molto).
dorme(p937,molto).
consumo_calorico(p937,alto).
fumare(p937,si).
persona(p412).
eta_protetta(p412,si).
sedentario(p412,no).
vita_sociale(p412,no).
tempo_social(p412, basso).
pasti(p412,troppi).
beve(p412,molto).
dorme(p412,molto).
consumo_calorico(p412,alto).
fumare(p412,si).
persona(p492).
eta_protetta(p492,si).
sedentario(p492,si).
vita_sociale(p492,si).
tempo_social(p492, basso).
pasti(p492,pochi).
beve(p492,molto).
dorme(p492,poco).
consumo_calorico(p492,alto).
fumare(p492,si).
persona(p769).
eta_protetta(p769,si).
sedentario(p769,si).
vita_sociale(p769,si).
tempo_social(p769, basso).
pasti(p769,pochi).
beve(p769,poco).
dorme(p769,poco).
consumo_calorico(p769,alto).
fumare(p769,si).
persona(p376).
eta_protetta(p376,si).
sedentario(p376,no).
vita_sociale(p376,no).
tempo_social(p376, basso).
pasti(p376,troppi).
beve(p376,molto).
dorme(p376,poco).
consumo_calorico(p376,alto).
fumare(p376,si).
persona(p26).
eta_protetta(p26,si).
sedentario(p26,no).
vita_sociale(p26,no).
tempo_social(p26, basso).
pasti(p26,troppi).
beve(p26,molto).
dorme(p26,poco).
consumo_calorico(p26,alto).
fumare(p26,si).
persona(p699).
eta_protetta(p699,si).
sedentario(p699,no).
vita_sociale(p699,no).
tempo_social(p699, basso).
pasti(p699,troppi).
beve(p699,molto).
dorme(p699,poco).
consumo_calorico(p699,alto).
fumare(p699,si).
persona(p462).
eta_protetta(p462,si).
sedentario(p462,no).
vita_sociale(p462,no).
tempo_social(p462, basso).
pasti(p462,troppi).
beve(p462,molto).
dorme(p462,poco).
consumo_calorico(p462,alto).
fumare(p462,si).
persona(p73).
eta_protetta(p73,si).
sedentario(p73,no).
vita_sociale(p73,no).
tempo_social(p73, basso).
pasti(p73,troppi).
beve(p73,molto).
dorme(p73,poco).
consumo_calorico(p73,alto).
fumare(p73,si).
persona(p375).
eta_protetta(p375,si).
sedentario(p375,no).
vita_sociale(p375,no).
tempo_social(p375, basso).
pasti(p375,troppi).
beve(p375,molto).
dorme(p375,poco).
consumo_calorico(p375,alto).
fumare(p375,si).
persona(p723).
eta_protetta(p723,si).
sedentario(p723,no).
vita_sociale(p723,no).
tempo_social(p723, basso).
pasti(p723,troppi).
beve(p723,molto).
dorme(p723,poco).
consumo_calorico(p723,alto).
fumare(p723,si).
persona(p833).
eta_protetta(p833,si).
sedentario(p833,si).
vita_sociale(p833,si).
beve(p833,poco).
dorme(p833,poco).
consumo_calorico(p833,alto).
fumare(p833,si).
persona(p963).
eta_protetta(p963,si).
sedentario(p963,si).
vita_sociale(p963,si).
beve(p963,poco).
dorme(p963,poco).
consumo_calorico(p963,alto).
fumare(p963,si).
persona(p229).
eta_protetta(p229,si).
sedentario(p229,si).
vita_sociale(p229,si).
beve(p229,poco).
dorme(p229,poco).
consumo_calorico(p229,alto).
fumare(p229,si).
persona(p963).
eta_protetta(p963,si).
sedentario(p963,si).
vita_sociale(p963,si).
beve(p963,poco).
dorme(p963,poco).
consumo_calorico(p963,alto).
fumare(p963,si).
persona(p105).
eta_protetta(p105,si).
sedentario(p105,si).
vita_sociale(p105,si).
beve(p105,poco).
dorme(p105,poco).
consumo_calorico(p105,alto).
fumare(p105,si).
persona(p697).
eta_protetta(p697,si).
sedentario(p697,si).
vita_sociale(p697,si).
beve(p697,poco).
dorme(p697,poco).
consumo_calorico(p697,alto).
fumare(p697,si).
persona(p589).
eta_protetta(p589,si).
sedentario(p589,si).
vita_sociale(p589,si).
beve(p589,poco).
dorme(p589,poco).
consumo_calorico(p589,alto).
fumare(p589,si).
persona(p746).
eta_protetta(p746,si).
sedentario(p746,si).
vita_sociale(p746,si).
beve(p746,poco).
dorme(p746,poco).
consumo_calorico(p746,alto).
fumare(p746,si).
persona(p181).
eta_protetta(p181,si).
sedentario(p181,si).
vita_sociale(p181,si).
beve(p181,poco).
dorme(p181,poco).
consumo_calorico(p181,alto).
fumare(p181,si).
persona(p461).
eta_protetta(p461,si).
sedentario(p461,si).
vita_sociale(p461,si).
beve(p461,poco).
dorme(p461,poco).
consumo_calorico(p461,alto).
fumare(p461,si).
persona(p486).
eta_protetta(p486,si).
sedentario(p486,si).
vita_sociale(p486,si).
beve(p486,poco).
dorme(p486,poco).
consumo_calorico(p486,alto).
fumare(p486,si).
persona(p78).
eta_protetta(p78,si).
sedentario(p78,si).
vita_sociale(p78,si).
beve(p78,poco).
dorme(p78,poco).
consumo_calorico(p78,alto).
fumare(p78,si).
persona(p482).
eta_protetta(p482,si).
sedentario(p482,si).
vita_sociale(p482,si).
beve(p482,poco).
dorme(p482,poco).
consumo_calorico(p482,alto).
fumare(p482,si).
persona(p582).
eta_protetta(p582,si).
sedentario(p582,si).
vita_sociale(p582,si).
beve(p582,poco).
dorme(p582,poco).
consumo_calorico(p582,alto).
fumare(p582,si).
persona(p981).
eta_protetta(p981,si).
sedentario(p981,si).
vita_sociale(p981,si).
beve(p981,poco).
dorme(p981,poco).
consumo_calorico(p981,alto).
fumare(p981,si).
persona(p3700).
eta_protetta(p3700,si).
sedentario(p3700,si).
vita_sociale(p3700,si).
tempo_social(p3700,basso).
pasti(p3700,troppi).
beve(p3700,molto).
dorme(p3700,molto).
consumo_calorico(p3700,alto).
fumare(p3700,si).
persona(p1388).
eta_protetta(p1388,si).
sedentario(p1388,si).
vita_sociale(p1388,si).
tempo_social(p1388,basso).
pasti(p1388,troppi).
beve(p1388,molto).
dorme(p1388,molto).
consumo_calorico(p1388,alto).
fumare(p1388,si).
persona(p3565).
eta_protetta(p3565,si).
sedentario(p3565,si).
vita_sociale(p3565,si).
tempo_social(p3565,basso).
pasti(p3565,troppi).
beve(p3565,molto).
dorme(p3565,molto).
consumo_calorico(p3565,alto).
fumare(p3565,si).
persona(p4632).
eta_protetta(p4632,si).
sedentario(p4632,si).
vita_sociale(p4632,si).
tempo_social(p4632,basso).
pasti(p4632,pochi).
beve(p4632,molto).
dorme(p4632,molto).
consumo_calorico(p4632,basso).
fumare(p4632,si).
persona(p9205).
eta_protetta(p9205,si).
sedentario(p9205,si).
vita_sociale(p9205,si).
tempo_social(p9205,basso).
pasti(p9205,pochi).
beve(p9205,molto).
dorme(p9205,molto).
consumo_calorico(p9205,basso).
fumare(p9205,si).
persona(p5285).
eta_protetta(p5285,si).
sedentario(p5285,no).
vita_sociale(p5285,no).
tempo_social(p5285,basso).
pasti(p5285,troppi).
beve(p5285,molto).
dorme(p5285,molto).
consumo_calorico(p5285,basso).
fumare(p5285,no).
persona(p5065).
eta_protetta(p5065,si).
sedentario(p5065,si).
vita_sociale(p5065,si).
tempo_social(p5065,basso).
pasti(p5065,troppi).
beve(p5065,molto).
dorme(p5065,molto).
consumo_calorico(p5065,alto).
fumare(p5065,si).
persona(p6505).
eta_protetta(p6505,si).
sedentario(p6505,si).
vita_sociale(p6505,si).
tempo_social(p6505,basso).
pasti(p6505,troppi).
beve(p6505,molto).
dorme(p6505,molto).
consumo_calorico(p6505,alto).
fumare(p6505,si).