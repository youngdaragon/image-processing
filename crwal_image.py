import requests
from selenium import webdriver
max_cnt=40
keyword='경치'
url=f'https://www.pexels.com/ko-kr/search/{keyword}/'

browser=webdriver.Chrome()
browser.maximize_window()
browser.get(url)

photo_items=browser.find_elements_by_class_name('photo-item__img')

img_urls=[x.get_attribute('data-big-src') for x in photo_items]
print(len(img_urls))
idx=1
for img_url in img_urls:
    browser.get(img_url)

    res=requests.get(img_url)
    if res.ok:
        file_name= f'{keyword}_{idx}.jpeg'
        with open(file_name,'wb') as f:
            f.write(res.content)
        print(f'({idx}) {file_name}')
        idx += 1
    
    if idx>max_cnt:
        break

browser.quit()