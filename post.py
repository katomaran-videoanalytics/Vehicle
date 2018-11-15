# importing the requests library
import requests
import time
import datetime

# defining the api-endpoint 
#incam:
#API_ENDPOINT = "http://159.89.172.250:4000/api/v1/vehicles/"
#outcam:
API_ENDPOINT = "http://159.89.172.250:4000/api/v1/vehicles/update_exit/"
s = datetime.datetime.now().strftime('%H:%M:%S')
# data to be sent to api
#incam:
'''
data ={"vehicle":{
"in_time":s,
"number_plate":"SBA1234H",
"vehicle_type":"normal"
}
}
'''
#outcam

data ={"vehicle":{
"out_time":s,
"number_plate":"SBA1234H",
}
}

# sending post request and saving response as response object
r = requests.put(url = API_ENDPOINT, json = data)
pastebin_url = r.text
print("The pastebin URL is:%s"%pastebin_url)
