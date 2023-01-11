import matplotlib.pyplot as plt

labels='E','I'
sizes=[10,90]
explode=(0.1,0.1)

fig1,ax1=plt.subplots()
ax1.pie(sizes, explode=explode,labels=labels,autopct='%1.1f%%',shadow=True, startangle=90)
ax1.axis('equal')
plt.show()