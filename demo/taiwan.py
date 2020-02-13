import time

import requests
from lxml import etree as et


def getdata(url):
    # url = "https://tw.51240.com/"
    res = requests.get(url)
    html = et.HTML(res.text)
    data = html.xpath('//div[@id="shuaxinshenfenzheng"]/table//table//text()')
    data = data[4:]
    return data


def xm(url):
    f1 = open(r"C:\Users\c\Desktop\name.txt", "w", encoding="utf8")
    while 1:
        # url = "https://tw.51240.com/"
        data = getdata(url)
        for i in range(int(len(data)/4)):
            # print(" ".join(data[i*4:(i*4) + 4]))
            line = " ".join(data[i*4:(i*4) + 4])
            f1.write(line+"\n")
            f1.flush()
    f1.close()


def address(url):
    headers = {
        'User-Agent': 'self-defind-user-agent',
        'Cookie': 'name=self-define-cookies-in header'
    }
    f1 = open(r"C:\Users\c\Desktop\address.txt", "w", encoding="utf8")
    index = 1
    while 1:
        res = requests.get(url+str(index), headers=headers)
        # print(res.text)
        html = et.HTML(res.text)
        data = html.xpath(
            '//div[@class="ui-branchs-warp"]//li//dd[@class="ui-branchs-address"]/span[@class="ui-branchs-info-sp"]//text()')
        print(data)
        for content in data:
            f1.write(content + "\n")
            f1.flush()
        index = index+1
    f1.close()


def company(url):
    headers = {
        'User-Agent': 'self-defind-user-agent'
    }

    f1 = open(r"C:\Users\c\Desktop\company.txt", "w", encoding="utf8")
    index = 0
    while 1:
        res = requests.get(url + str(index), headers=headers)
        time.sleep(1)
        html = et.HTML(res.text)
        # data = html.xpath('//div[@class="view-content"]//a/text()')
        data = html.xpath('//div[@class="view-content"]/div/div/h2/a/text()')
        print(data)
        if len(data) == 0:
            break
        for content in data:
            f1.write(content + "\n")
            f1.flush()
        index = index + 1
    f1.close()

def company_one(url):
    headers = {
        'User-Agent': 'self-defind-user-agent'
    }

    f1 = open(r"C:\Users\c\Desktop\company1.txt", "w", encoding="utf8")
    index = 1
    while 1:
        if index == 1:
            uri = url + '.html'
        else:
            uri = url + '_p' + str(index) + '.html'
        print(uri)
        # https://www.71ab.com/province_32_p2.html
        res = requests.get(uri, headers=headers)
        # time.sleep(1)
        html = et.HTML(res.text)
        # data = html.xpath('//div[@class="view-content"]//a/text()')
        # data = html.xpath('//ul[@class="list-item"]/li/h3/a/text()')
        data = html.xpath('//strong[@class="px14"]/text()')
        print(data)
        if len(data) == 0:
            break
        for content in data:
            f1.write(content + "\n")
            f1.flush()
        index = index + 1
    f1.close()


if __name__ == "__main__":
    # url = "https://tw.51240.com/"
    # xm(url)
    # url = 'https://www.p2peye.com/yhwd/p16/page'
    # dz(url)
    # url = "https://waizi.mingluji.com/taxonomy/term/825?page="
    # comp(url)
    url = "https://www.71ab.com/province_32"
    company_one(url)
