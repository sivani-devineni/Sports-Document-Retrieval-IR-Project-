import requests
from bs4 import BeautifulSoup
import time

def page_info(page_url, index):
    
    page_data = requests.get(page_url)
    
    soup = BeautifulSoup(page_data.content, "html.parser")

    title = soup.find("h1", {"itemprop":"headline"}).get_text()

    posting_body = soup.find("p", {"itemprop" : "articleBody"}).get_text()

    with open(f"./dataset/{index}.txt", 'w', encoding='utf-8') as f:
        print(title.strip(), file=f)
        print(posting_body.strip(), file=f)
        f.close()

        
def field_search(page_url, page_number):

    page_url = page_url + "page-{}".format(page_number) 
    page = requests.get(page_url)

    soup = BeautifulSoup(page.content, "html.parser")

    all_links = soup.find_all("h3", {'itemprop': 'headline'})
    Links = []
    for link in all_links:
        Links.append(link.find("a")['href'])


    index = (page_number-1)*24+1
    for Link in Links:
        page_info(Link,index)
        index += 1
    



count = 148
page_url = f"https://www.shortpedia.com/en-in/politics-news/"


while(count < 250):
   field_search(page_url, count)
   count+=1
