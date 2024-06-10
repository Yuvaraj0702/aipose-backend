mappings = {
    "questions": [
        {
            "category": "seated_posture",
            "items": [
                {
                    "id": "1",
                    "scenarios": {
                        "negative": {
                            "risk": "Medium",
                            "affected_parts": ["Lower back"],
                            "conditions": ["Lower back pain", "Herniated disk"],
                            "current": "You're sitting back but slouching in your chair.",
                            "recommendation": "Sit fully back in your chair, ensuring the backrest supports your spine."
                        },
                        "positive": {
                            "risk": "High",
                            "affected_parts": ["Lower back"],
                            "conditions": ["Lower back pain", "Herniated disk"],
                            "current": "You're leaning forward, leaving your back unsupported.",
                            "recommendation": "Sit fully back in your chair, ensuring the backrest supports your spine."
                        },
                        "neutral": {
                            "risk": "Low",
                            "affected_parts": ["Lower back"],
                            "conditions": [""],
                            "current": "Your seating position keeps your back adequately supported.",
                            "recommendation": ""
                        }
                    }
                },
                {
                    "id": "2",
                    "scenarios": {
                        "negative": {
                            "risk": "High",
                            "affected_parts": ["Knees", "Lower back"],
                            "conditions": ["Knee pain", "Lower back pain", "Herniated disk"],
                            "current": "You are sitting too low. This puts pressure on the spine as it is being compressed.",
                            "recommendation": "Sit high such that your hips are slightly higher than your knees with feet supported. Adjust your chair or sit on top of a cushion to get the desired height. If your feet are hanging, use a footrest or box for support."
                        },
                        "positive": {
                            "risk": "Medium",
                            "affected_parts": ["Lower back"],
                            "conditions": ["Lower back pain", "Herniated disk (with prolonged sitting)"],
                            "current": "While sitting with hips in-line with the knees creates a 90° angle, it is not the right angle for you!",
                            "recommendation": "Sit high such that your hips are slightly higher than your knees with feet supported. Adjust your chair or sit on top of a cushion to get the desired height. If your feet are hanging, use a footrest or box for support."
                        },
                        "neutral": {
                            "risk": "Low",
                            "affected_parts": ["Lower back"],
                            "conditions": [""],
                            "current": "You are sitting at a good height to reduce pressure on the spine. Ensure your feet are well supported.",
                            "recommendation": ""
                        }
                    }
                }
            ]
        },
        {
            "category": "hand_position",
            "items": [
                {
                    "id": "1",
                    "scenarios": {
                        "negative": {
                            "risk": "Medium",
                            "affected_parts": ["Wrist"],
                            "conditions": ["Tendonitis"],
                            "current": "Your wrists flexed downwards while typing, it can strain the tendons at the back of your hand, increasing the risk of discomfort and repetitive strain injuries.",
                            "recommendation": "Ensure your keyboard is positioned about a palm's distance from the edge of the table, maintaining a neutral posture. This setup reduces wrist stress and encourages better posture. Keeping your wrists in a neutral position while typing ensures minimal strain, promoting long-term comfort and reducing the risk of repetitive stress injuries."
                        },
                        "positive": {
                            "risk": "Medium",
                            "affected_parts": ["Wrist"],
                            "conditions": ["Carpal tunnel syndrome"],
                            "current": "Your wrists are flexed upwards while typing, it can increase tension in the tendons and nerves, potentially leading to conditions like carpal tunnel syndrome.",
                            "recommendation": "Ensure your keyboard is positioned about a palm's distance from the edge of the table, maintaining a neutral posture. This setup reduces wrist stress and encourages better posture. Keeping your wrists in a neutral position while typing ensures minimal strain, promoting long-term comfort and reducing the risk of repetitive stress injuries."
                        },
                        "neutral": {
                            "risk": "Low",
                            "affected_parts": ["Wrist"],
                            "conditions": [""],
                            "current": "Your wrists in a neutral position while typing ensures minimal strain, promoting long-term comfort and reducing the risk of repetitive stress injuries.",
                            "recommendation": ""
                        }
                    }
                },
                {
                    "id": "2",
                    "scenarios": {
                        "negative": {
                            "risk": "Medium",
                            "affected_parts": ["Wrist"],
                            "conditions": ["Tendonitis"],
                            "current": "Your current grip makes an arched positioning of the fingers, this can lead to finger strain over extended periods, especially if pressing the buttons with force.",
                            "recommendation": "Use an ergonomically designed keyboard that fits comfortably in your hand. Consider using a mouse pad with a wrist rest to support your hand and wrist. Take regular breaks, stretch your fingers, and practice a relaxed hand grip to reduce strain and enhance comfort."
                        },
                        "positive": {
                            "risk": "Medium",
                            "affected_parts": ["Hand"],
                            "conditions": ["Tennis elbow"],
                            "current": "Your current way of mousinging can lead to finger fatigue quicker since you're using only your fingertips .",
                            "recommendation": "Use an ergonomically designed mouse that fits comfortably in your hand. Consider using a mouse pad with a wrist rest to support your hand and wrist. Take regular breaks, stretch your fingers, and practice a relaxed hand grip to reduce strain and enhance comfort."
                        },
                        "neutral": {
                            "risk": "Low",
                            "affected_parts": ["Wrist"],
                            "conditions": [""],
                            "current": "Your wrists in a neutral position while using the keyboard ensures minimal strain, promoting long-term comfort and reducing the risk of repetitive stress injuries.",
                            "recommendation": ""
                        }
                    }
                },
                {
                    "id": "3",
                    "scenarios": {
                        "negative": {
                            "risk": "High",
                            "affected_parts": ["Wrist"],
                            "conditions": ["Carpal tunnel syndrome"],
                            "current": "Your wrists are twisted while using keyboard, it can lead to uneven pressure on the wrist structures, potentially causing discomfort and increasing the likelihood of wrist-related ailments.",
                            "recommendation": "Ensure your keyboard is positioned about a palm's distance from the edge of the table, maintaining a neutral posture. This setup reduces wrist stress and encourages better posture. Keeping your wrists in a neutral position while using the mouse ensures minimal strain, promoting long-term comfort and reducing the risk of repetitive stress injuries."
                        },
                        "positive": {
                            "risk": "High",
                            "affected_parts": ["Wrist"],
                            "conditions": ["Carpal tunnel syndrome"],
                            "current": "Your wrists are twisted while using the keyboard, it can lead to uneven pressure on the wrist structures, potentially causing discomfort and increasing the likelihood of wrist-related ailments.",
                            "recommendation": "Ensure your keyboard is positioned about a palm's distance from the edge of the table, maintaining a neutral posture. This setup reduces wrist stress and encourages better posture. Keeping your wrists in a neutral position while using the mouse ensures minimal strain, promoting long-term comfort and reducing the risk of repetitive stress injuries."
                        },
                        "neutral": {
                            "risk": "Low",
                            "affected_parts": ["Wrist"],
                            "conditions": [""],
                            "current": "Your wrists in a neutral position while using the mouse ensures minimal strain, promoting long-term comfort and reducing the risk of repetitive stress injuries.",
                            "recommendation": ""
                        }
                    }
                }

            ]
        },
        {
            "category": "desk_position",
            "items": [
                {
                    "id": "1",
                    "scenarios": {
                        "negative": {
                            "risk": "High",
                            "affected_parts": ["Shoulders", "Neck", "Upper back", "Arms"],
                            "conditions": ["Cervical spondylosis", "Thoracic outlet syndrome"],
                            "current": "Your desk is too high; elbows are low, lifting arms and shoulders during typing.",
                            "recommendation": "Align your desk height with your elbow to ensure your arms and shoulders remain relaxed during work."
                        },
                        "positive": {
                            "risk": "High",
                            "affected_parts": ["Shoulders", "Neck", "Upper back", "Arms"],
                            "conditions": ["Cervical spondylosis", "Thoracic outlet syndrome"],
                            "current": "Your desk is too low; elbows are high, causing a forward lean to type.",
                            "recommendation": "Align your desk height with your elbow to ensure your arms and shoulders remain relaxed during work."
                        },
                        "neutral": {
                            "risk": "Low",
                            "affected_parts": ["Lower back"],
                            "conditions": [""],
                            "current": "Your desk is at the right height; elbows align, arms and shoulders relaxed.",
                            "recommendation": ""
                        }
                    }
                },
                {
                    "id": "2",
                    "scenarios": {
                        "negative": {
                            "risk": "Low",
                            "affected_parts": ["Eyes", "Neck", "Head"],
                            "conditions": ["Eye strain", "Neck strain", "Migraine"],
                            "current": "Your screen is too close, leading to potential eye strain and excessive head and neck movement.",
                            "recommendation": "Position your laptop an arm's length away. Bring it closer only if reading becomes difficult."
                        },
                        "positive": {
                            "risk": "High",
                            "affected_parts": ["Shoulders", "Lower back", "Upper back", "Arms"],
                            "conditions": ["Cervical spondylosis", "Thoracic outlet syndrome", "Lower back pain", "Herniated disc"],
                            "current": "Your screen is too far, causing a tendency to lean in.",
                            "recommendation": "Position your laptop an arm's length away. Bring it closer only if reading becomes difficult."
                        },
                        "neutral": {
                            "risk": "Low",
                            "affected_parts": ["Lower back"],
                            "conditions": [""],
                            "current": "You have an optimal viewing distance; use an external keyboard and mouse for relaxed arms and shoulders.",
                            "recommendation": ""
                        }
                    }
                },
                {
                    "id": "3",
                    "scenarios": {
                        "negative": {
                            "risk": "High",
                            "affected_parts": ["Neck", "Shoulders", "Upper back", "Lower back"],
                            "conditions": ["Cervical spondylosis", "Thoracic outlet syndrome", "Lower back pain", "Herniated disc"],
                            "current": "Laptop is too low; you lean forward, losing back support and risking upper back and shoulder pain.",
                            "recommendation": "Adjust your laptop using a stand or books so the first line of text aligns with eye level. This alignment helps prevent neck and upper back strain."
                        },
                        "positive": {
                            "risk": "Medium",
                            "affected_parts": ["Neck", "Head"],
                            "conditions": ["Cervical spondylosis"],
                            "current": "Laptop is too high; you tilt your head up, tightening neck muscles and increasing the risk of cervical spondylosis.",
                            "recommendation": "Adjust your laptop using a stand or books so the first line of text aligns with eye level. This alignment helps prevent neck and upper back strain."
                        },
                        "neutral": {
                            "risk": "Low",
                            "affected_parts": ["Neck"],
                            "conditions": [""],
                            "current": "Your laptop is at eye level, promoting good posture and minimizing neck and upper back strain.",
                            "recommendation": ""
                        }
                    }
                }
            ]
        }
    ]
}