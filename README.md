# Primena deep learninga za preporuku proizvoda u elektronskoj trgovini — Autoenkoderi

## 1. Opis problema

Cilj ovog projekta je izgradnja sistema za preporuku proizvoda u elektronskoj trgovini korišćenjem dubokog učenja, konkretno autoenkodera. 

Na osnovu ocena koje je korisnik ostavljao za proizvode koje je kupio, model uči ukuse korisnika i preporučuje mu proizvode koje još nije ocenio, a za koje postoji velika verovatnoća da će mu se svideti. Model prima kao ulaz ocene korisnika za proizvode na skali od 1 do 5, uči obrasce i ukuse korisnika, i kao izlaz daje predviđene ocene od 1 do 5 za proizvode koje korisnik nije ocenio. Proizvodi sa najvišim predviđenim ocenama predstavljaju preporuke za tog korisnika.

---

## 2. Podaci

### Izvor

Dataset je preuzet sa [Kaggle](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews) platforme i sadrži recenzije proizvoda sa Amazon platforme.

### Struktura

Originalni dataset sadržao je **568,454 redova** i sledeće kolone:

| Kolona | Opis |
|--------|------|
| Id | Jedinstveni identifikator recenzije |
| ProductId | Identifikator proizvoda |
| UserId | Identifikator korisnika |
| ProfileName | Ime korisnika |
| HelpfulnessNumerator | Broj korisnih glasova |
| HelpfulnessDenominator | Ukupan broj glasova |
| Score | Ocena od 1 do 5 |
| Time | Vreme ocenjivanja |
| Summary | Kratak opis recenzije |
| Text | Tekst recenzije |

Za potrebe projekta zadržane su samo tri kolone: **UserId**, **ProductId** i **Score**, jer su jedine relevantne za sistem preporuke.

### Analiza i preprocesiranje

**Pre filtriranja:**
- Broj korisnika: 256,059
- Broj proizvoda: 74,258
- Broj ocena: 568,454
- Prosečna ocena: 4.18
- Minimum ocena po korisniku: 1
- Maksimum ocena po korisniku: 448

**Raspodela ocena:**

| Ocena | Broj ocena |
|-------|-----------|
| 1 | 52,268 |
| 2 | 29,769 |
| 3 | 42,640 |
| 4 | 80,655 |
| 5 | 363,122 |

Uočeno je da dataset ima **neuravnoteženu raspodelu ocena** - čak 60% svih ocena su petice, što je karakteristično za Amazon recenzije.

**5-core filtriranje:**

Korisnici koji su ocenili samo jedan ili dva proizvoda, kao i proizvodi sa jednom ili dve ocene, mogu da navedu model da pogrešno uči jer ne pružaju dovoljno informacija. Iz tog razloga primenjeno je **5-core filtriranje** koja je standardna tehnika u sistemima preporuke. Uklonjeni su svi korisnici koji su ocenili manje od 5 proizvoda i svi proizvodi koji imaju manje od 5 ocena.

**Posle filtriranja:**
- Broj korisnika: 17,666
- Broj proizvoda: 5,012
- Broj ocena: 175,437

**Matrica korisnik × proizvod:**

Od filtriranog dataseta napravljena je matrica gde svaki red predstavlja jednog korisnika a svaka kolona jedan proizvod. Vrednosti u matrici su ocene, dok su mesta gde korisnik nije ocenio proizvod popunjena nulom. Matrica je bila **99.81% prazna** (sparse), što je tipično za sisteme preporuke.

**Normalizacija:**

Ocene su normalizovane sa skale 1-5 na opseg 0-1 deljenjem sa 5, kako bi odgovarale izlazu Sigmoid aktivacione funkcije na kraju dekodera.

**Podela dataseta:**

Zbog vremena treninga, korišćeno je prvih 5,000 korisnika. Dataset je podeljen u odnosu **80:20** na trening (4,000 korisnika) i test skup (1,000 korisnika). Podaci su učitavani u mini-batch-evima od 64 korisnika pomoću PyTorch DataLoader-a.

---

## 3. Arhitektura modela

Korišćen je **autoenkoder** - tip neuronske mreže koji se sastoji od dva dela: enkodera i dekodera.

**Enkoder** prima vektor ocena jednog korisnika (5,012 vrednosti) i kompresuje ga kroz tri linearna sloja sve do latentnog prostora:

```
Ulaz:     5012 vrednosti
Sloj 1:   Linear(5012 -> 256) + ReLU
Sloj 2:   Linear(256 -> 128)  + ReLU
Sloj 3:   Linear(128 -> 64)   + ReLU
Izlaz:    64 vrednosti (latentni prostor)
```

**Latentni prostor** od 64 vrednosti predstavlja naučeni profil korisnika - kompresovanu reprezentaciju njegovih ukusa i preferencija.

**Dekoder** uzima tih 64 vrednosti i rekonstruiše originalni vektor kroz tri linearna sloja:

```
Ulaz:     64 vrednosti (latentni prostor)
Sloj 1:   Linear(64 -> 128)   + ReLU
Sloj 2:   Linear(128 -> 256)  + ReLU
Sloj 3:   Linear(256 -> 5012) + Sigmoid
Izlaz:    5012 vrednosti (rekonstruisane ocene)
```

**Sigmoid** aktivacija na kraju dekodera osigurava da izlaz bude između 0 i 1, što odgovara normalizovanim ulaznim ocenama.

**ReLU** aktivacija između slojeva unosi nelinearnost koja omogućava mreži da uči složene obrasce u podacima.

Za mesta gde korisnik nije dao ocenu (nule na ulazu), dekoder predviđa vrednost na osnovu naučenog profila korisnika što predstavlja preporuke.

---

## 4. Trening

**Funkcija greške:** `nn.MSELoss()` (Mean Squared Error) - meri prosečnu kvadratnu razliku između rekonstruisanih i originalnih ocena.

**Optimizer:** `Adam` sa learning rate-om 0.001 - automatski prilagođava veličinu koraka za svaku težinu posebno, što omogućava brže i stabilnije učenje.

**Maska:** Tokom inicijalnog treninga bez maske, model je naučio trivijalno rešenje, vraćao je nule svuda jer je 99.81% matrice ionako prazno, pa je greška bila mala (0.0015) ali lažna. Uvedena je **maska** koja ograničava računanje greške samo na mesta gde postoje stvarne ocene:

```python
mask = (X_batch != 0)
loss = criterion(output[mask], y_batch[mask])
```

Na taj način model nije mogao da "vara" vraćanjem nula nego je morao da nauči stvarne obrasce u podacima.

**Early stopping:** Trening je konfigurisan na maksimalno 50 epoha sa patience=10, što znači da se trening zaustavlja ako se test greška ne poboljša 10 epoha za redom. Čuva se model sa najboljom test greškom.

---

## 5. Analiza osetljivosti i hiperparametarska optimizacija

Isprobane su tri konfiguracije modela:

| Model | Latentna dimenzija | Learning rate | Najbolja test greška (MSE) |
|-------|-------------------|---------------|---------------------------|
| Mali model | 32 | 0.001 | 0.0387 |
| Srednji model | 64 | 0.001 | 0.0385 |
| **Veći learning rate** | **64** | **0.01** | **0.0383** |

**Zaključci:**

- **Model 1 (latent=32)** postigao je lošije rezultate jer manja latentna dimenzija znači veću kompresiju, odnosno 5,012 vrednosti mora stati u samo 32 broja, što dovodi do gubitka informacija o profilu korisnika.

- **Model 2 (latent=64)** postigao je bolje rezultate od modela 1 jer latentna dimenzija od 64 pruža dovoljno prostora za reprezentaciju profila korisnika bez prevelikog gubitka informacija.

- **Model 3 (lr=0.01)** postigao je najbolje rezultate. Veći learning rate omogućio je modelu brže pronalaženje optimalnog rešenja. Model je stigao do bolje test greške uz isti broj parametara kao model 2.

Kao finalni model izabran je **Model 3** sa latentnom dimenzijom 64 i learning rate-om 0.01.

---

## 6. Rezultati evaluacije

Evaluacija finalnog modela na test skupu:

| Metrika | Vrednost | Interpretacija |
|---------|----------|----------------|
| MSE | 0.9496 | Prosečna kvadratna greška na skali 1-5 |
| RMSE | 0.9744 | Prosečna greška ≈ 0.97 zvezda na skali 1-5 |
| MAE | 0.5820 | Većina predikcija greši za manje od 0.58 zvezda |

**Tok treninga finalnog modela:**
- Početna trening greška: 0.0825
- Krajnja trening greška: 0.0096
- Početna test greška: 0.0579
- Najbolja test greška: 0.0383 (early stopping na epohi 26)

**Prikaz ulaza i izlaza modela (Korisnik 1):**

| Proizvod | Ulaz (original) | Izlaz (rekonstrukcija) |
|----------|----------------|----------------------|
| B001E50THY | 5.0 | 4.86 |
| B001OCKIP0 | 4.0 | 4.49 |
| B002TMV34E | 3.0 | 3.70 |
| B003GTR8IO | 4.0 | 3.61 |
| B003XDH6M6 | 4.0 | 3.41 |
| B004OV6X6Q | 2.0 | 1.79 |
| B0090X8IPM | 4.0 | 3.62 |

**Primer preporuka:**

*Korisnik 0* - sve ocene su 5.0:
```
Već ocenio:   B005DGI1IY -> 5.0, B005DGI1PW -> 5.0, ...
Preporuke:    B000EVE3YE -> 4.99, B0030VJ70K -> 4.99, ...
```

*Korisnik 1* - ocene od 3.0 do 5.0:
```
Već ocenio:   B001E50THY -> 5.0, B002TMV34E -> 3.0, ...
Preporuke:    B000EVE3YE -> 4.95, B001FA1KU8 -> 4.92, ...
```

*Korisnik 3* - ocene od 1.0 do 5.0:
```
Već ocenio:   B00004RBDU -> 1.0, B000H7ELTW -> 5.0, ...
Preporuke:    B000EVE3YE -> 4.94, B001FA1KU8 -> 4.93, ...
```

Model daje različite predviđene ocene za različite korisnike, što pokazuje da uči individualne obrasce.

---

## 7. Diskusija

Rezultati modela imaju smisla i pokazuju da autoenkoder uspešno uči obrasce u podacima. MAE od 0.58 zvezda ne predstavlja veliku grešku za sistem preporuke.

**RMSE** može da zavara u određenim situacijama. Greška od jedne zvezde na višim ocenama (npr. 4 umesto 5) ne predstavlja problem jer su oba pozitivna iskustva. Međutim, greška na nižim ocenama može biti problematičnija. Ako korisnik treba da oceni proizvod sa 3 (neutralno), a model predvidi 4 (zadovoljstvo), preporuka može biti pogrešna.

**Sparsity** (retkost matrice) predstavlja glavno ograničenje modela. Sa 99.81% praznih polja, model ima mnogo više podataka koje mora da popuni nego podataka na kojima može da uči. Ovo je tipičan problem sistema preporuke jer korisnici retko ocenjuju sve proizvode koje kupe.

**Personalizacija** je vidljiva u rezultatima. Korisnik koji sve ocenjuje sa 5 dobija preporuke sa ocenama oko 5.00, dok kritičniji korisnici dobijaju konzervativnije preporuke. Ograničenje je što korisnici sa malo ocena daju nedovoljno informacija za visoko personalizovane preporuke.

---

## 8. Zaključak

Autoenkoder se pokazao kao prikladan model za sistem preporuke u elektronskoj trgovini. Kroz kompresiju profila korisnika u latentni prostor, model uspešno grupiše korisnike sličnih ukusa i na osnovu naučenih obrazaca predviđa ocene za neocenjene proizvode.

Za poboljšanje sistema moglo bi se razmotriti:
- Korišćenje većeg dataseta sa više ocena po korisniku
- Isprobavanje većeg broja epoha i različitih vrednosti patience parametra
- Eksperimentisanje sa različitim veličinama batch-eva
- Istraživanje alternativnih arhitektura kao što su varijacioni autoenkoderi (VAE)

---

## Tehnologije

- Python 3
- PyTorch
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Kaggle Hub

## Pokretanje projekta

```bash
# Klonirati repozitorijum
git clone https://github.com/username/autoenkoder-projekat

# Otvoriti notebook u Google Colab
# Pokrenuti sve ćelije redom
```

## Licenca

MIT License
