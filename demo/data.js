const DEMO_DATA = {
  "dataset": "redial",
  "k": 10,
  "aggregator": "bi-GRU aggregator (39.9M params, frozen)",
  "adaptCkpt": "logs/adapt/redial/rnn/add_bias_True_mul_bias_True/version_0/checkpoints/best.ckpt",
  "conversations": [
    {
      "convId": "22945",
      "turnId": "2294518",
      "turns": [
        {
          "speaker": "SYSTEM",
          "text": "Hi! how are you"
        },
        {
          "speaker": "USER",
          "text": "hey, how are you I'm looking for war and action movies got any reccomedations?"
        },
        {
          "speaker": "SYSTEM",
          "text": "Sure Great Action movie will be Inception (2010)"
        },
        {
          "speaker": "USER",
          "text": "I seen that and loved it"
        },
        {
          "speaker": "SYSTEM",
          "text": "Also have you seen Avengers: Infinity War (2018) ?!"
        },
        {
          "speaker": "USER",
          "text": "I have not I'll check it out my parent sae that last night"
        },
        {
          "speaker": "SYSTEM",
          "text": "Yes its pretty good as well as Black Panther (2018) Really how they like it?!"
        },
        {
          "speaker": "USER",
          "text": "they loved it any other recomendations?"
        }
      ],
      "groundTruth": [
        "American Sniper",
        "Jarhead (film)"
      ],
      "before": [
        {
          "name": "The Dark Knight (film)",
          "token": 739,
          "score": 2393.042,
          "gt": false
        },
        {
          "name": "The Matrix (club)",
          "token": 2005,
          "score": 2392.635,
          "gt": false
        },
        {
          "name": "Mad Max: Fury Road",
          "token": 909,
          "score": 2392.279,
          "gt": false
        },
        {
          "name": "The Bourne Identity (2002 film)",
          "token": 1877,
          "score": 2391.3,
          "gt": false
        },
        {
          "name": "The Avengers (2012 film)",
          "token": 1040,
          "score": 2389.805,
          "gt": false
        },
        {
          "name": "Inception",
          "token": 2,
          "score": 2389.707,
          "gt": false
        },
        {
          "name": "Interstellar (film)",
          "token": 190,
          "score": 2388.634,
          "gt": false
        },
        {
          "name": "John Wick (film)",
          "token": 1019,
          "score": 2388.162,
          "gt": false
        },
        {
          "name": "Gladiator (1992 film)",
          "token": 1737,
          "score": 2387.753,
          "gt": false
        },
        {
          "name": "Gladiator (2000 film)",
          "token": 683,
          "score": 2387.346,
          "gt": false
        }
      ],
      "after": [
        {
          "name": "Sherlock Holmes (2010 film)",
          "token": 5978,
          "score": 2643.171,
          "gt": false
        },
        {
          "name": "Mr. & Mrs. Smith (2005 film)",
          "token": 896,
          "score": 2642.998,
          "gt": false
        },
        {
          "name": "Captain America: Civil War",
          "token": 1242,
          "score": 2642.713,
          "gt": false
        },
        {
          "name": "American Sniper",
          "token": 153,
          "score": 2642.699,
          "gt": true
        },
        {
          "name": "Deadpool (film)",
          "token": 294,
          "score": 2642.498,
          "gt": false
        },
        {
          "name": "Mad Max (film)",
          "token": 1081,
          "score": 2642.2,
          "gt": false
        },
        {
          "name": "Black Panther (film)",
          "token": 61,
          "score": 2641.468,
          "gt": false
        },
        {
          "name": "Easter Parade (film)",
          "token": 2651,
          "score": 2641.331,
          "gt": false
        },
        {
          "name": "Bicentennial Man (film)",
          "token": 4430,
          "score": 2641.283,
          "gt": false
        },
        {
          "name": "Life (2017 film)",
          "token": 369,
          "score": 2641.246,
          "gt": false
        }
      ],
      "gtRankBefore": 729,
      "gtRankAfter": 4
    },
    {
      "convId": "20737",
      "turnId": "2073715",
      "turns": [
        {
          "speaker": "SYSTEM",
          "text": "Hey there! How are you?"
        },
        {
          "speaker": "USER",
          "text": "I'm good thanks. Do you have any comedies you could recommend?"
        },
        {
          "speaker": "SYSTEM",
          "text": "Yes have you seen Jumanji (2017)"
        },
        {
          "speaker": "USER",
          "text": "No, I have not seen Jumanji (2017) I will have to add that one to my list."
        },
        {
          "speaker": "SYSTEM",
          "text": "Great! Its pretty funny What about Down to Earth (2001) I watched it the other night. I forgot how much I liked it!"
        },
        {
          "speaker": "USER",
          "text": "I have never heard of Down to Earth (2001) I will add that to my list and hopefully see it this wekend. What about action movies?"
        },
        {
          "speaker": "SYSTEM",
          "text": "Chris Rock is in it!"
        },
        {
          "speaker": "USER",
          "text": "I love Chris Rock, he is a pretty funny guy."
        }
      ],
      "groundTruth": [
        "Deadpool (film)",
        "Deadpool 2"
      ],
      "before": [
        {
          "name": "Jumanji",
          "token": 848,
          "score": 2679.35,
          "gt": false
        },
        {
          "name": "The 40-Year-Old Virgin",
          "token": 2268,
          "score": 2672.905,
          "gt": false
        },
        {
          "name": "Kevin Hart: What Now%3F",
          "token": 4695,
          "score": 2672.125,
          "gt": false
        },
        {
          "name": "22 Jump Street",
          "token": 1567,
          "score": 2671.915,
          "gt": false
        },
        {
          "name": "The Hangover (film series)",
          "token": 1792,
          "score": 2671.745,
          "gt": false
        },
        {
          "name": "Why Him%3F",
          "token": 4261,
          "score": 2671.212,
          "gt": false
        },
        {
          "name": "Bridesmaids (1989 film)",
          "token": 4228,
          "score": 2670.892,
          "gt": false
        },
        {
          "name": "Ip Man 3",
          "token": 5208,
          "score": 2670.883,
          "gt": false
        },
        {
          "name": "Crazy Heart",
          "token": 2057,
          "score": 2670.329,
          "gt": false
        },
        {
          "name": "Due Date",
          "token": 916,
          "score": 2670.2,
          "gt": false
        }
      ],
      "after": [
        {
          "name": "Why Him%3F",
          "token": 4261,
          "score": 2960.479,
          "gt": false
        },
        {
          "name": "Thor: Ragnarok",
          "token": 1104,
          "score": 2959.583,
          "gt": false
        },
        {
          "name": "The Boss Baby",
          "token": 245,
          "score": 2958.768,
          "gt": false
        },
        {
          "name": "Game Night (film)",
          "token": 1439,
          "score": 2958.734,
          "gt": false
        },
        {
          "name": "Grown Ups (film)",
          "token": 868,
          "score": 2958.242,
          "gt": false
        },
        {
          "name": "Deadpool (film)",
          "token": 294,
          "score": 2958.18,
          "gt": true
        },
        {
          "name": "Spider-Man: Homecoming",
          "token": 1658,
          "score": 2957.976,
          "gt": false
        },
        {
          "name": "Sherlock Holmes (2010 film)",
          "token": 5978,
          "score": 2957.938,
          "gt": false
        },
        {
          "name": "Mr. & Mrs. Smith (2005 film)",
          "token": 896,
          "score": 2957.863,
          "gt": false
        },
        {
          "name": "The Wedding Date",
          "token": 146,
          "score": 2957.821,
          "gt": false
        }
      ],
      "gtRankBefore": 348,
      "gtRankAfter": 6
    },
    {
      "convId": "22326",
      "turnId": "2232611",
      "turns": [
        {
          "speaker": "SYSTEM",
          "text": "Good Morning Are you looking for a certain type of movie?"
        },
        {
          "speaker": "USER",
          "text": "Hello good morning"
        },
        {
          "speaker": "SYSTEM",
          "text": "I just saw Early Man, it was pretty cute."
        },
        {
          "speaker": "USER",
          "text": "oh i like crimes do you habe some recommendations?"
        },
        {
          "speaker": "SYSTEM",
          "text": "oh, have you seen Making a Murderer on Netflix? It's long, but very interesting"
        },
        {
          "speaker": "USER",
          "text": "i have seen that one i like it"
        }
      ],
      "groundTruth": [
        "Murder on the Orient Express (2017 film)"
      ],
      "before": [
        {
          "name": "Zootopia",
          "token": 523,
          "score": 2366.042,
          "gt": false
        },
        {
          "name": "The Secret Life of Pets",
          "token": 3717,
          "score": 2365.688,
          "gt": false
        },
        {
          "name": "The Lego Movie",
          "token": 383,
          "score": 2365.362,
          "gt": false
        },
        {
          "name": "The Grand Budapest Hotel",
          "token": 1158,
          "score": 2364.818,
          "gt": false
        },
        {
          "name": "The Lego Batman Movie",
          "token": 780,
          "score": 2364.794,
          "gt": false
        },
        {
          "name": "Despicable Me (film)",
          "token": 1009,
          "score": 2364.592,
          "gt": false
        },
        {
          "name": "The Nightmare Before Christmas",
          "token": 1046,
          "score": 2364.1,
          "gt": false
        },
        {
          "name": "The Incredibles (film score)",
          "token": 639,
          "score": 2363.781,
          "gt": false
        },
        {
          "name": "Goodfellas",
          "token": 336,
          "score": 2363.742,
          "gt": false
        },
        {
          "name": "Sherlock Holmes (1922 film)",
          "token": 3402,
          "score": 2363.693,
          "gt": false
        }
      ],
      "after": [
        {
          "name": "Sherlock Holmes (2010 film)",
          "token": 5978,
          "score": 2624.652,
          "gt": false
        },
        {
          "name": "Mr. & Mrs. Smith (2005 film)",
          "token": 896,
          "score": 2620.139,
          "gt": false
        },
        {
          "name": "Game Night (film)",
          "token": 1439,
          "score": 2619.854,
          "gt": false
        },
        {
          "name": "Deadpool (film)",
          "token": 294,
          "score": 2619.786,
          "gt": false
        },
        {
          "name": "Bicentennial Man (film)",
          "token": 4430,
          "score": 2619.153,
          "gt": false
        },
        {
          "name": "Trainwreck (film)",
          "token": 1394,
          "score": 2618.798,
          "gt": false
        },
        {
          "name": "King Boxer",
          "token": 1915,
          "score": 2618.342,
          "gt": false
        },
        {
          "name": "Murder on the Orient Express (2017 film)",
          "token": 370,
          "score": 2618.191,
          "gt": true
        },
        {
          "name": "Sweeney Todd: The Demon Barber of Fleet Street (2007 film)",
          "token": 4020,
          "score": 2618.085,
          "gt": false
        },
        {
          "name": "Goodfellas",
          "token": 336,
          "score": 2617.855,
          "gt": false
        }
      ],
      "gtRankBefore": 307,
      "gtRankAfter": 8
    }
  ]
};
