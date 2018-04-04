#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created by C.L.Wang
import json
import time
import urllib
import urllib2

from project_utils import show_string, batch, time_elapsed

app_key = ''
secret_key = ''


def get_access_token(app_key, secret_key):
    api_key_url = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=%s&client_secret=%s'

    # client_id 为官网获取的AK， client_secret 为官网获取的SK
    host = (api_key_url % (app_key, secret_key))
    request = urllib2.Request(host)
    request.add_header('Content-Type', 'application/json; charset=UTF-8')
    response = urllib2.urlopen(request)
    content = response.read()
    keys = json.loads(content)
    access_token = keys['access_token']

    return access_token


def recognize_img_words(access_token, img):
    ocr_url = 'https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic/?access_token=%s'
    url = (ocr_url % access_token)

    # 上传的参数
    data = dict()
    data['languagetype'] = "CHN_ENG"  # 识别文字

    # 图片数据源，网络图片或本地图片
    if img.startswith('http://'):
        data['url'] = img
    else:
        image_data = open(img, 'rb').read()
        data['image'] = image_data.encode('base64').replace('\n', '')

    # 发送请求
    decoded_data = urllib.urlencode(data)
    req = urllib2.Request(url, data=decoded_data)
    req.add_header("Content-Type", "application/x-www-form-urlencoded")

    # 获取请求的数据，并读取内容
    resp = urllib2.urlopen(req)
    content = resp.read()

    # 识别出的图片数据
    words_result = json.loads(content)['words_result']
    words_list = list()
    for words in words_result:
        words_list.append(words['words'])
    return words_list


if __name__ == '__main__':
    local_img = './data/text_img2.jpeg'
    online_img = "http://www.zhaoniupai.com/hbv/upload/20150714_LiangDuiBan.jpg"

    s_time = time.time()

    access_token = get_access_token(app_key=app_key, secret_key=secret_key)
    print 'access_token: %s' % access_token

    print '\nwords_list:'
    words_list = recognize_img_words(access_token, online_img)
    for x in batch(words_list, 5):  # 每次打印5个数据
        show_string(x)

    print "time elapsed: %s" % time_elapsed(s_time, time.time())
