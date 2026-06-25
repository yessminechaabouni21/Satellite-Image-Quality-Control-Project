import requests

r = requests.post(
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
    "/protocol/openid-connect/token",
    data={
        "client_id": "cdse-public",
        "grant_type": "password",
        "username": "chaabouniyessmine3@gmail.com",
        "password": "y42$uKM3hCyw.eN",  
    },
    timeout=30,
)
print(r.status_code)
print(r.json())   # read the error_description field