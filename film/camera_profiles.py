"""
Radiance Camera Profiles
------------------------
Database of real-world cinema cameras, lenses, and formats for the Radiance Manager node.
"""

# =============================================================================
# CINEMA CAMERAS
# =============================================================================

CAMERAS = {
    "ARRI": {
        "Alexa 65 (IMAX)": {
            "sensor": "Open Gate A3X CMOS",
            "size": "54.12 x 25.58 mm",
            "resolution": "6.5K",
            "dynamic_range": "14+ stops",
            "native_iso": 800,
            "mount": "XPL",
        },
        "Alexa Mini LF": {
            "sensor": "Large Format ARRI ALEV III (A2X) CMOS",
            "size": "36.70 x 25.54 mm",
            "resolution": "4.5K",
            "dynamic_range": "14+ stops",
            "native_iso": 800,
            "mount": "LPL",
            "notes": "Compact / handheld / drone body. Same A2X sensor as Alexa LF.",
        },
        "Alexa 35": {
            "sensor": "Super 35 ARRI ALEV 4 CMOS",
            "size": "27.99 x 19.22 mm",
            "resolution": "4.6K",
            "dynamic_range": "17 stops",
            "native_iso": 800,
            "mount": "LPL",
        },
        "Alexa LF": {
            "sensor": "Large Format ARRI ALEV III (A2X) CMOS",
            "size": "36.70 x 25.54 mm",
            "resolution": "4.5K",
            "dynamic_range": "14+ stops",
            "native_iso": 800,
            "mount": "LPL",
            "notes": "Full-size studio body. Same A2X sensor as Alexa Mini LF but larger form factor, built-in ND, and MVF-2 viewfinder.",
        },
        "Amira": {
            "sensor": "Super 35 ARRI ALEV III CMOS",
            "size": "28.17 x 18.13 mm",
            "resolution": "3.2K",
            "dynamic_range": "14+ stops",
            "native_iso": 800,
            "mount": "PL",
        },
    },
    "RED": {
        "V-Raptor XL 8K VV": {
            "sensor": "Vista Vision CMOS",
            "size": "40.96 x 21.60 mm",
            "resolution": "8K",
            "dynamic_range": "17+ stops",
            "native_iso": 800,
            "mount": "PL",
        },
        "Monstro 8K VV": {
            "sensor": "Vista Vision CMOS",
            "size": "40.96 x 21.60 mm",
            "resolution": "8K",
            "dynamic_range": "17+ stops",
            "native_iso": 800,
            "mount": "PL",
        },
        "Komodo-X": {
            "sensor": "Super 35 CMOS",
            "size": "27.03 x 14.26 mm",
            "resolution": "6K",
            "dynamic_range": "16+ stops",
            "native_iso": 800,
            "mount": "RF",
        },
        "Helium 8K S35": {
            "sensor": "Super 35 CMOS",
            "size": "29.90 x 15.77 mm",
            "resolution": "8K",
            "dynamic_range": "16.5+ stops",
            "native_iso": 800,
            "mount": "PL",
        },
        "Gemini 5K S35": {
            "sensor": "Super 35 CMOS",
            "size": "30.72 x 18.00 mm",
            "resolution": "5K",
            "dynamic_range": "16.5+ stops",
            "native_iso": 800,
            "mount": "PL",
        },
    },
    "Sony": {
        "Venice 2 8K": {
            "sensor": "Full Frame CMOS",
            "size": "36.2 x 24.1 mm",
            "resolution": "8.6K",
            "dynamic_range": "16 stops",
            "native_iso": 800,
            "mount": "PL/E",
        },
        "Venice 6K": {
            "sensor": "Full Frame CMOS",
            "size": "36.2 x 24.1 mm",
            "resolution": "6K",
            "dynamic_range": "15+ stops",
            "native_iso": 500,
            "mount": "PL/E",
        },
        "FX9": {
            "sensor": "Full Frame 6K Exmor R CMOS",
            "size": "35.7 x 18.8 mm",
            "resolution": "6K",
            "dynamic_range": "15+ stops",
            "native_iso": 800,
            "mount": "E-mount",
        },
        "FX6": {
            "sensor": "Full Frame 4K Exmor R CMOS",
            "size": "35.6 x 23.8 mm",
            "resolution": "4K",
            "dynamic_range": "15+ stops",
            "native_iso": 800,
            "mount": "E-mount",
        },
    },
    "Panavision": {
        "Millennium DXL2": {
            "sensor": "Large Format RED Monstro 8K VV",
            "size": "40.96 x 21.60 mm",
            "resolution": "8K",
            "dynamic_range": "16 stops",
            "native_iso": 1600,  # Panavision's recommended EI for their colour science / LUT stack.
            # Underlying RED Monstro sensor native ISO is 800.
            "mount": "PV 70",
        },
        "Panaflex Millennium XL2 (Film)": {
            "sensor": "35mm Film Gate",
            "size": "24.89 x 18.67 mm",
            "resolution": "Analog (approx 6K)",
            "dynamic_range": "14 stops (Vision3)",
            "native_iso": 500,
            "mount": "PV",
        },
    },
    "IMAX": {
        "IMAX MKIV (15/70mm Film)": {
            "sensor": "15-perf 70mm Film",
            "size": "70.41 x 52.63 mm",
            "resolution": "Analog (approx 18K horizontal)",
            "dynamic_range": "15+ stops",
            "native_iso": 50,  # Daylight balanced stock
            "mount": "IMAX",
        }
    },
    "Blackmagic": {
        "URSA Mini Pro 12K": {
            "sensor": "Super 35 CMOS",
            "size": "27.03 x 14.25 mm",
            "resolution": "12K",
            "dynamic_range": "14 stops",
            "native_iso": 800,
            "mount": "PL",
        },
        "Cinema Camera 6K": {
            "sensor": "Full Frame CMOS",
            "size": "36 x 24 mm",
            "resolution": "6K",
            "dynamic_range": "13 stops",
            "native_iso": 400,
            "mount": "L-Mount",
        },
    },
}


# =============================================================================
# CINEMA LENSES
# =============================================================================

LENSES = {
    "ARRI": {
        "Signature Primes": {
            "type": "Prime",
            "focal_lengths": [
                "12mm",
                "15mm",
                "18mm",
                "21mm",
                "25mm",
                "29mm",
                "35mm",
                "40mm",
                "47mm",
                "58mm",
                "75mm",
                "95mm",
                "125mm",
                "150mm",
                "200mm",
                "280mm",
            ],
            "aperture": "T1.8",
            "character": "Modern, clean, creamy bokeh, natural skin tones",
        },
        "Master Primes": {
            "type": "Prime",
            "focal_lengths": [
                "12mm",
                "14mm",
                "16mm",
                "18mm",
                "21mm",
                "25mm",
                "27mm",
                "32mm",
                "35mm",
                "40mm",
                "50mm",
                "65mm",
                "75mm",
                "100mm",
                "135mm",
                "150mm",
            ],
            "aperture": "T1.3",
            "character": "Extremely sharp, high contrast, zero breathing",
        },
        "Ultra Primes": {
            "type": "Prime",
            "focal_lengths": [
                "8R",
                "10mm",
                "12mm",
                "14mm",
                "16mm",
                "20mm",
                "24mm",
                "28mm",
                "32mm",
                "40mm",
                "50mm",
                "65mm",
                "85mm",
                "100mm",
                "135mm",
                "180mm",
            ],
            "aperture": "T1.9",
            "character": "Contrast, color matched, compact",
        },
        "Master Anamorphic": {
            "type": "Anamorphic",
            "focal_lengths": [
                "28mm",
                "35mm",
                "40mm",
                "50mm",
                "60mm",
                "75mm",
                "100mm",
                "135mm",
                "180mm",
            ],
            "aperture": "T1.9",
            "character": "Low distortion, cinematic flares, oval bokeh",
        },
    },
    "Cooke": {
        "S7/i Full Frame": {
            "type": "Prime",
            "focal_lengths": [
                "16mm",
                "18mm",
                "21mm",
                "25mm",
                "27mm",
                "32mm",
                "40mm",
                "50mm",
                "65mm",
                "75mm",
                "100mm",
                "135mm",
            ],
            "aperture": "T2.0",
            "character": "The 'Cooke Look', warm, smooth falloff, pleasing skin tones",
        },
        "Anamorphic /i": {
            "type": "Anamorphic",
            "focal_lengths": [
                "25mm",
                "32mm",
                "40mm",
                "50mm",
                "65mm",
                "75mm",
                "100mm",
                "135mm",
                "180mm",
                "300mm",
            ],
            "aperture": "T2.3",
            "character": "Classic anamorphic, oval bokeh, 'Cooke Look'",
        },
        "Panchro/i Classic": {
            "type": "Vintage Prime",
            "focal_lengths": [
                "18mm",
                "21mm",
                "25mm",
                "27mm",
                "32mm",
                "40mm",
                "50mm",
                "65mm",
                "75mm",
                "100mm",
                "135mm",
                "152mm",
            ],
            "aperture": "T2.2",
            "character": "Vintage warmth, gentle sharpness, classic bokeh",
        },
    },
    "Zeiss": {
        "Supreme Primes": {
            "type": "Prime",
            "focal_lengths": [
                "15mm",
                "18mm",
                "21mm",
                "25mm",
                "29mm",
                "35mm",
                "50mm",
                "65mm",
                "85mm",
                "100mm",
                "135mm",
                "150mm",
                "200mm",
            ],
            "aperture": "T1.5",
            "character": "Versatile, gentle sharpness, smooth transition",
        },
        "CP.3 Compact Primes": {
            "type": "Prime",
            "focal_lengths": [
                "15mm",
                "18mm",
                "21mm",
                "25mm",
                "28mm",
                "35mm",
                "50mm",
                "85mm",
                "100mm",
                "135mm",
            ],
            "aperture": "T2.9",
            "character": "Clean, neutral, color matched",
        },
    },
    "Panavision": {
        "Primo 70": {
            "type": "Prime",
            "focal_lengths": [
                "27mm",
                "35mm",
                "40mm",
                "50mm",
                "65mm",
                "80mm",
                "100mm",
                "125mm",
                "150mm",
                "200mm",
                "250mm",
            ],
            "aperture": "T2.0",
            "character": "Optimized for large format, high resolution, classic Panavision look",
        },
        "C-Series Anamorphic": {
            "type": "Anamorphic",
            "focal_lengths": [
                "20mm",
                "30mm",
                "35mm",
                "40mm",
                "50mm",
                "60mm",
                "75mm",
                "100mm",
                "180mm",
            ],
            "aperture": "T2.8",
            "character": "Vintage, blue flares, organic imperfections, classic Hollywood",
        },
        "E-Series Anamorphic": {
            "type": "Anamorphic",
            "focal_lengths": [
                "28mm",
                "35mm",
                "40mm",
                "50mm",
                "75mm",
                "85mm",
                "100mm",
                "135mm",
                "180mm",
            ],
            "aperture": "T2.0",
            "character": "Sharper than C-Series, refined coatings, classic bokeh",
        },
    },
    "Leica": {
        "Summilux-C": {
            "type": "Prime",
            "focal_lengths": [
                "16mm",
                "18mm",
                "21mm",
                "25mm",
                "29mm",
                "35mm",
                "40mm",
                "50mm",
                "65mm",
                "75mm",
                "100mm",
                "135mm",
            ],
            "aperture": "T1.4",
            "character": "Humanistic, creamy out-of-focus, sharp but not clinical",
        },
        "Thalia": {
            "type": "Prime",
            "focal_lengths": [
                "24mm",
                "30mm",
                "35mm",
                "45mm",
                "55mm",
                "70mm",
                "100mm",
                "120mm",
                "180mm",
            ],
            "aperture": "T2.2",
            "character": "Dimensionality, cinematic look, large image circle",
        },
    },
    "Canon": {
        "K-35 (Vintage)": {
            "type": "Vintage Prime",
            "focal_lengths": ["18mm", "24mm", "35mm", "55mm", "85mm"],
            "aperture": "T1.3",
            "character": "Vintage flares, low contrast, dreamy skin tones (Aliens, Barry Lyndon)",
        }
    },
    "Angenieux": {
        "Optimo Ultra 12x": {
            "type": "Zoom",
            "focal_lengths": ["24-290mm"],
            "aperture": "T2.8",
            "character": "The industry standard zoom, perfect color matching, cinematic feel",
        },
        "EZ Series": {
            "type": "Zoom",
            "focal_lengths": ["15-40mm", "30-90mm", "45-135mm"],
            "aperture": "T2.0",
            "character": "Fast, versatile, modern look",
        },
    },
}

# =============================================================================
# APERTURES
# =============================================================================

APERTURES = [
    "T0.95 (Dreamy / Razor Thin)",
    "T1.3 (Master Prime Open)",
    "T1.4 (Very Shallow)",
    "T1.8 (Standard Fast Prime)",
    "T2.0 (Cinematic Separation)",
    "T2.8 (Standard Open)",
    "T4.0 (Balanced)",
    "T5.6 (Deep Focus)",
    "T8.0 (Sharp Landscape)",
    "T11 (Diffraction Limit)",
    "T16 (Sunstars)",
    "T22 (Everything in Focus)",
]

# =============================================================================
# SHUTTER ANGLES
# =============================================================================

SHUTTER_ANGLES = [
    "180° (Standard Motion - 1/48s)",
    "90° (Action / Staccato - 1/96s)",
    "45° (Saving Private Ryan - 1/192s)",
    "270° (Dreamy Blur - 1/32s)",
    "360° (Maximum Blur - 1/24s)",
    "172.8° (1/50s Flicker Free)",
    "144° (1/60s Flicker Free)",
]

# =============================================================================
# ISO SETTINGS
# =============================================================================

ISO_SETTINGS = [
    "50 ISO (Fine Grain)",
    "100 ISO (Clean Daylight)",
    "200 ISO (Standard Day)",
    "400 ISO (Standard)",
    "800 ISO (Native Digital)",
    "1600 ISO (Low Light)",
    "3200 ISO (High Sensitivity)",
    "6400 ISO (Grainy / Noise)",
    "12800 ISO (Night Vision)",
]
