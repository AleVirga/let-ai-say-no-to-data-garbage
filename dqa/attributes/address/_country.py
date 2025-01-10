
import pycountry

# list of cch countries iso2 codes
C_COUNTRIES = [
    "AM",
    "AT",
    "BA",
    "BG",
    "CH",
    "CY",
    "CZ",
    "EE",
    "EG",
    "GB",
    "GR",
    "HR",
    "HU",
    "IE",
    "IT",
    "KV",
    "LT",
    "LV",
    "MD",
    "ME",
    "NG",
    "PL",
    "RO",
    "RS",
    "SI",
    "SK",
    "UA",
]

COUNTRIES_DICT = {
    c.alpha_2: c.name for c in pycountry.countries if c.alpha_2 in C_COUNTRIES
}

COUNTRIES_DICT["KV"] = "Kosovo"
COUNTRIES = dict(sorted(COUNTRIES_DICT.items()))
