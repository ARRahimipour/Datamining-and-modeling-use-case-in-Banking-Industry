# پروژه خوشه‌بندی مشتریان با استفاده از KMeans

این پروژه یک الگوریتم خوشه‌بندی برای گروه‌بندی مشتریان بر اساس ویژگی‌هایی نظیر سن، درآمد، میزان وام و تعداد تراکنش‌ها استفاده می‌کند. هدف این است که خوشه‌هایی با ویژگی‌های مشابه شناسایی شوند تا بتوان تحلیل‌های تجاری نظیر شناسایی خوشه‌هایی با وام‌های بالاتر را انجام داد.

---

## مراحل اجرای کد

### مرحله 1: ایجاد داده‌های شبیه‌سازی شده

در این بخش، داده‌هایی شبیه‌سازی شده برای 100 نفر مشتری ساخته می‌شود. ویژگی‌های مشتریان شامل سن، درآمد، میزان وام و تعداد تراکنش‌ها هستند. داده‌ها در یک DataFrame به نام `df` ذخیره می‌شوند.

```python
import pandas as pd
import numpy as np

# 1. Data generation (assuming data is ready and loaded into the df variable)

# Age, income, and loan amount are defined in integers (no decimals)

data = {
    'age': np.random.randint(20, 60, 100),  # Age without decimals
    'income': np.random.randint(30000000, 100000000, 100),  # Income in Toman
    'loan_amount': np.random.randint(5000000, 40000000, 100),  # Loan amount in Toman
    'transaction_count': np.random.randint(10, 100, 100)  # Number of transactions
}

df = pd.DataFrame(data)
```

### مرحله 2: پیش‌پردازش داده‌ها و استانداردسازی (Standardization)

در این مرحله، داده‌ها با استفاده از `StandardScaler` استاندارد می‌شوند تا مقیاس ویژگی‌ها تأثیری بر الگوریتم KMeans نگذارد.

```python
from sklearn.preprocessing import StandardScaler

# 2. Data preprocessing

# Standardizing the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
```

### مرحله 3: تعیین تعداد بهینه خوشه‌ها با استفاده از silhouette_score

در این بخش، برای تعیین تعداد بهینه خوشه‌ها از شاخص `silhouette_score` استفاده می‌شود. این شاخص نشان می‌دهد که چقدر خوشه‌ها از یکدیگر جدا هستند و چقدر داده‌های داخل هر خوشه به یکدیگر شباهت دارند.

```python
from sklearn.metrics import silhouette_score

# 3. Determining the optimal number of clusters using silhouette score

silhouette_scores = []
k_values = range(2, 10)  # Trying different values of k (clusters)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(scaled_data)
    score = silhouette_score(scaled_data, labels)
    silhouette_scores.append(score)

# Select the number of clusters with the highest silhouette score
best_k = k_values[np.argmax(silhouette_scores)]
print(f"Optimal number of clusters: {best_k}")
```

### مرحله 4: اجرای خوشه‌بندی با تعداد خوشه‌های بهینه

پس از تعیین تعداد بهینه خوشه‌ها، الگوریتم KMeans با آن تعداد خوشه‌ها اجرا می‌شود و برچسب خوشه‌ها به داده‌ها اضافه می‌شود.

```python
from sklearn.cluster import KMeans

# 4. Clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=best_k, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled_data)
```

### مرحله 5: تحلیل نتایج

در این بخش، میانگین میزان وام در هر خوشه محاسبه می‌شود. سپس خوشه‌ای که بیشترین میانگین وام را دارد به عنوان خوشه هدف شناسایی می‌شود.

```python
# 5. Analyzing the results

# To calculate Lift, identify the cluster with the highest average loan amount as the target cluster
lift_values = df.groupby('cluster')['loan_amount'].mean()
target_cluster = lift_values.idxmax()
print(f"Target cluster for high loan likelihood: {target_cluster}")
print("Lift values for each cluster:")
print(lift_values / df['loan_amount'].mean())
```

### مرحله 6: نمایش نتایج به صورت جدول

در این مرحله، میانگین ویژگی‌ها (سن، درآمد، وام و تعداد تراکنش‌ها) برای هر خوشه محاسبه و در قالب یک جدول نمایش داده می‌شود.

```python
# 6. Displaying results as integers in a formatted table
result = df.groupby('cluster').mean().round(0).astype(int)
print(result.to_string(index=True, justify="center", col_space=10, header=True))
```

---

## توضیحات

1. **تعداد خوشه‌های بهینه:** تعداد خوشه‌ها بر اساس بالاترین مقدار `silhouette_score` تعیین می‌شود.
2. **خوشه هدف (Target Cluster):** خوشه‌ای که بیشترین میانگین وام را دارد، به عنوان خوشه هدف برای احتمال دریافت وام بالاتر انتخاب می‌شود.
3. **شاخص Lift:** برای هر خوشه، میانگین وام به میانگین کلی وام‌ها تقسیم می‌شود. این شاخص نشان می‌دهد که خوشه‌ها تا چه حد از نظر وام برجسته‌تر از میانگین کل هستند.
4. **جدول میانگین ویژگی‌ها:** میانگین ویژگی‌های مختلف (سن، درآمد، وام و تعداد تراکنش‌ها) برای هر خوشه محاسبه و نمایش داده می‌شود.

---

## نصب پیش‌نیازها

برای اجرای این کد، نیاز به نصب بسته‌های زیر دارید:

```bash
pip install pandas numpy scikit-learn
```

---

## نتیجه نهایی

خروجی این کد شامل موارد زیر خواهد بود:

1. تعداد خوشه‌های بهینه: تعداد خوشه‌ای که بالاترین مقدار `silhouette_score` را دارد.
2. خوشه هدف (Target Cluster): خوشه‌ای که بیشترین میانگین وام را دارد.
3. شاخص Lift: مقادیر شاخص Lift برای هر خوشه.
4. جدول میانگین ویژگی‌ها: نمایش میانگین ویژگی‌های مختلف برای هر خوشه.

این فرآیند می‌تواند به کسب‌وکارها کمک کند تا مشتریانی که احتمال دریافت وام بالاتری دارند را شناسایی کنند یا به طور کلی برای تحلیل‌های بازاریابی و تصمیم‌گیری‌های تجاری مورد استفاده قرار گیرد.
