import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud


data  = pd.read_csv("4.csv")

# Boxplot for `total_bill`
sns.boxplot(x=data['total_bill'])
plt.title('Boxplot of Total Bill')
plt.show()
# Boxplot for `tip`
sns.boxplot(x=data['tip'])
plt.title('Boxplot of Tip')
plt.show()


# Histogram for `total_bill`
plt.hist(data['total_bill'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Total Bill')
plt.xlabel('Total Bill')
plt.ylabel('Frequency')
plt.show()

# Histogram for `tip`
plt.hist(data['tip'], bins=20, color='green', edgecolor='black')
plt.title('Distribution of Tip')
plt.xlabel('Tip')
plt.ylabel('Frequency')
plt.show()


# Compute correlation matrix
corr_matrix = data[['total_bill', 'tip', 'size']].corr()
# Heatmap for correlation matrix
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# Bar chart for average `total_bill` by `day`
sns.barplot(x='day', y='total_bill', data=data, palette='Blues')
plt.title('Average Total Bill by Day')
plt.show()
# Pie chart for gender distribution
gender_count = data['sex'].value_counts()
gender_count.plot.pie(autopct='%1.1f%%', colors=['lightblue', 'pink'], startangle=90, figsize=(6, 6))
plt.title('Gender Distribution')
plt.ylabel('')
plt.show()



# Assuming 'day' and 'sex' are categories, and 'total_bill' is the value
fig = px.sunburst(data, path=['day', 'sex'], values='total_bill', title="Tree Map of Total Bill by Day and Gender")
fig.show()


# Generating Word Cloud for 'time' column
time_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(data['time']))
plt.figure(figsize=(10, 6))
plt.imshow(time_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Time')
plt.axis('off')
plt.show()
