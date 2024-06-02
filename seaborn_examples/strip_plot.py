import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="white", color_codes=True)

tips = sns.load_dataset("tips")
#sns.stripplot(data=tips, x="total_bill", y="day")
sns.stripplot(data=tips, x="total_bill", y="day", hue="day", legend=False)


plt.show()
