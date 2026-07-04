# Walmart Recruiting — Store Sales Forecasting

Kaggle competition-ის ([link](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting))
გუნდური ფინალური პროექტი. მიზანია Walmart-ის მაღაზიების დეპარტამენტების
ყოველკვირეული გაყიდვების (`Weekly_Sales`) პროგნოზირება და სხვადასხვა
Time-Series არქიტექტურის შედარება.

## შეფასების მეტრიკა — WMAE

კომპეტიცია იყენებს **Weighted Mean Absolute Error**-ს, სადაც სადღესასწაულო
კვირები (`IsHoliday=True`) ფასდება 5-ჯერ მეტი წონით:

```
WMAE = (1 / Σ wᵢ) · Σ wᵢ · |yᵢ − ŷᵢ|,   wᵢ = 5 თუ holiday, სხვა შემთხვევაში 1
```

იმპლემენტაცია: [`src/metrics.py`](src/metrics.py).

## მონაცემები

| ფაილი | აღწერა |
|-------|--------|
| `train.csv` | Store, Dept, Date, Weekly_Sales, IsHoliday (2010-02 → 2012-10) |
| `test.csv` | Store, Dept, Date, IsHoliday (2012-11 → 2013-07) |
| `features.csv` | Temperature, Fuel_Price, MarkDown1–5, CPI, Unemployment |
| `stores.csv` | Store, Type (A/B/C), Size |
| `sampleSubmission.csv` | `Id = Store_Dept_Date`, Weekly_Sales |

45 მაღაზია, 81 დეპარტამენტი, ~3300 (Store, Dept) დროითი მწკრივი, თითო ~143 კვირა.

### EDA-ს ძირითადი დასკვნები
- ძლიერი კვირეული სეზონურობა — Thanksgiving (კვირა 47) და Christmas (კვირა 51) პიკები.
- `Type`/`Size`/`Dept` გაყიდვების მთავარი დრაივერებია.
- MarkDown1–5 არ არსებობს 2011 ნოემბრამდე → იმპუტაცია 0-ით ("აქცია არ იყო").
- ~1285 უარყოფითი გაყიდვა (დაბრუნებები) → პროგნოზი იჭრება 0-ზე submission-ისთვის.

## რეპოზიტორიის სტრუქტურა

```
├── src/                      # საზიარო კოდი (ორივე გუნდის წევრი იყენებს)
│   ├── config.py             # გარემოს დეტექცია (Kaggle/local), ბილიკები, credential-ები
│   ├── data_loader.py        # raw ფაილების წაკითხვა და merge
│   ├── features.py           # feature engineering (კალენდარი, markdown, lag-ები)
│   ├── metrics.py            # WMAE
│   ├── validation.py         # time-series CV (holdout / expanding window)
│   ├── pipeline.py           # sklearn Pipeline — უშუალოდ raw test-ზე ეშვება
│   ├── submission.py         # Kaggle submission-ის აწყობა
│   └── tracking.py           # MLflow / DagsHub setup
├── notebooks/
│   ├── 01_EDA.ipynb                      # საერთო EDA
│   ├── model_experiment_TEMPLATE.ipynb   # თარგი თითო მოდელისთვის
│   ├── model_experiment_{Model}.ipynb    # თითო არქიტექტურა (იხ. ქვემოთ)
│   └── model_inference.ipynb             # საუკეთესო მოდელი Registry-დან → submission
├── requirements.txt          # ბაზისური გარემო
├── requirements-models.txt   # XGBoost, LightGBM, statsmodels, Prophet
└── requirements-dl.txt       # torch, neuralforecast, TimesFM
```

## გარემოს მომზადება

### ლოკალურად
```bash
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -r requirements.txt
cp .env.example .env        # და შეავსე credential-ები
```

### Kaggle-ზე
თითო notebook იწყება **bootstrap** უჯრით, რომელიც:
1. ამოწმებს `/kaggle/input`-ს (Kaggle vs local),
2. Kaggle-ზე კლონავს ამ რეპოს `src/` პაკეტის მისაღებად,
3. მონაცემებს კითხულობს `/kaggle/input/walmart-recruiting-store-sales-forecasting/`-დან.

Credential-ები Kaggle-ზე ინახება **Add-ons → Secrets**-ში (იგივე სახელებით,
რაც `.env`-ში). ინტერნეტი უნდა იყოს ჩართული (Settings → Internet on).

## MLflow-ის სტრუქტურა

Experiment tracking → DagsHub-ის MLflow. თითო არქიტექტურას აქვს ცალკე
**experiment** `{Model}_Training`, რომლის შიგნით run-ებია:

- `{Model}_Cleaning` — მონაცემების გაწმენდის გადაწყვეტილებები
- `{Model}_Feature_Selection` — შერჩეული ფიჩერები
- `{Model}_CV` — expanding-window backtest, WMAE თითო fold-ზე
- `{Model}_Final` — საბოლოო fit + Pipeline-ის დალოგვა

თითო არქიტექტურის საუკეთესო მოდელი ინახება **Pipeline**-ად (ეშვება raw test-ზე);
საერთო საუკეთესო კი რეგისტრირდება **Model Registry**-ში სახელით `walmart_best_model`,
საიდანაც `model_inference.ipynb` ტვირთავს მას პროგნოზისთვის.

## მოდელების განაწილება (4 + 4)

| # | არქიტექტურა | კატეგორია | პასუხისმგებელი | სტატუსი |
|---|-------------|-----------|----------------|---------|
| 1 | TFT | Deep Learning | Sopho | ⬜ |
| 2 | N-BEATS | Deep Learning | Sopho | ⬜ |
| 3 | ARIMA | Classical | Sopho | ⬜ |
| 4 | SARIMA | Classical | Sopho | ⬜ |
| 5 | XGBoost | Tree-based | Teammate | ⬜ |
| 6 | LightGBM | Tree-based | Teammate | ⬜ |
| 7 | DLinear | Deep Learning | Teammate | ⬜ |
| 8 | Prophet | Classical | Teammate | ⬜ |
| 9 | TimesFM *(bonus)* | Foundation | – | ⬜ |

> თითო მოდელი კეთდება `model_experiment_TEMPLATE.ipynb`-ის კოპირებით.

## შედეგები (ივსება მოდელების დასრულებისას)

| მოდელი | CV WMAE | Holdout WMAE | Kaggle Public | Kaggle Private |
|--------|---------|--------------|---------------|----------------|
| TFT | – | – | – | – |
| N-BEATS | – | – | – | – |
| ARIMA | – | – | – | – |
| SARIMA | – | – | – | – |
| XGBoost | – | – | – | – |
| LightGBM | – | – | – | – |
| DLinear | – | – | – | – |
| Prophet | – | – | – | – |
| TimesFM | – | – | – | – |

## სამუშაო პროცესი (workflow)

1. **ერთად:** EDA + feature engineering (`01_EDA.ipynb`, `src/`).
2. **ცალ-ცალკე:** თითოეული ვიღებთ 4 მოდელს, ვქმნით `model_experiment_{Model}.ipynb`-ს.
3. თითო მოდელი = ცალკე git commit (`feat: XGBoost experiment` და ა.შ.).
4. საუკეთესო მოდელს ვარეგისტრირებთ Registry-ში და ვაგენერირებთ submission-ს.
5. README-ს შედეგების ცხრილს ვავსებთ პარალელურად.
