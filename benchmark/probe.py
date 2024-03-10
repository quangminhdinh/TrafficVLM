from .utils import convert_dict_to_desc_list
from .metrics import batch_evaluate


gt = {
   "20230707_8_SN46_T1":[
      {
         "labels":[
            "4"
         ],
         "caption_pedestrian":"The pedestrian stands still on the left side behind the vehicle, his body positioned perpendicular to it and slightly leaning towards the right. With a close distance from the car, his line of sight focuses on the vehicle itself. Although he has almost noticed it, his awareness of the vehicle is not complete. In his 30s, the man is of male gender and stands at a height of 170 cm. He is wearing a brown jacket on his upper body and navy blue slacks on his lower body. The weather is clear with bright brightness, and the road surface conditions are dry on the level asphalt road. The traffic volume is usual, and it is a residential road with two-way traffic. However, there are no sidewalks or roadside strips on both sides of the road. The pedestrian's general action is standing still, but there is also a special action of rushing out, albeit slowly.",
         "caption_vehicle":"The vehicle was positioned diagonally to the left in front of the pedestrian, and the relative distance between the two was close. The vehicle's field of view indicated that the pedestrian was not visible to the driver. Unfortunately, the vehicle collided with the pedestrian while moving at a speed of 20 km/h. Meanwhile, the environment conditions revealed that the pedestrian was a male in his 30s, standing at a height of 170 cm. He was wearing a brown jacket and navy blue slacks. The weather was clear and bright, and the road surface was dry and level. The road itself was a residential one with two-way traffic and no sidewalks present on both sides. Additionally, the road surface was made of asphalt. The traffic volume was usual, and there were no roadside strips on both sides.",
         "start_time":"39.395",
         "end_time":"44.663"
      },
      {
         "labels":[
            "3"
         ],
         "caption_pedestrian":"The pedestrian is a male in his 30s with a height of 170 cm. He is wearing a brown jacket on his upper body and navy blue slacks on his lower body. The weather is clear and the brightness is bright. The road surface conditions are dry and the road is level asphalt. The traffic volume is usual, and it is a residential road with two-way traffic and no sidewalks on both sides or roadside strips. The pedestrian's body is positioned perpendicular to the vehicle and to the right side of it. He is close to the vehicle and notices its presence. The pedestrian is standing still but suddenly rushes out. He moves slowly towards the right direction.",
         "caption_vehicle":"The vehicle is positioned on the left side of the pedestrian and is fairly close to them. The pedestrian is within the vehicle's field of view, so they are visible to the driver. The vehicle's current action is to continue moving straight ahead, and it is traveling at a speed of 20 km/h. The environment conditions surrounding the event involve a male pedestrian in his 30s with a height of 170 cm. He is wearing a brown jacket on the upper body and navy blue slacks on the lower body. The weather is clear, with bright visibility. The road surface is dry and level, made of asphalt. The traffic volume is usual, and the road where the event is taking place is a residential road with two-way traffic. There is no sidewalk on both sides of the road, and there is no roadside strip present.Overall, the vehicle is in close proximity to the pedestrian and has clear visibility of them. The environment conditions, such as the weather and road surface, are optimal for driving.",
         "start_time":"36.162",
         "end_time":"39.395"
      },
      {
         "labels":[
            "2"
         ],
         "caption_pedestrian":"The pedestrian, a male in his 30s, stood near the vehicle with his body perpendicular and to the right of it. He wore a brown jacket on his upper body and navy blue slacks on his lower body. The weather was clear, and the brightness of the surroundings was bright. The road surface was dry, and the road was level, with asphalt. The residential road had two-way traffic with no sidewalk on both sides, nor a roadside strip. The pedestrian seemed unaware of the vehicle and was about to rush out in front of it. The event took place in usual traffic conditions.",
         "caption_vehicle":"The vehicle is positioned on the right side of the pedestrian at a near distance. The pedestrian is visible within the vehicle's field of view. The vehicle is going straight ahead at a speed of 20 km/h. The environment conditions include a male in his 30s with a height of 170 cm. He is wearing a brown jacket and navy blue slacks. The weather is clear, with a bright brightness level. The road surface conditions are dry and level, with asphalt as the road surface type. The traffic volume is usual on this residential road with two-way traffic. There is no sidewalk on both sides, and there is no roadside strip on both sides.",
         "start_time":"34.725",
         "end_time":"36.162"
      },
      {
         "labels":[
            "1"
         ],
         "caption_pedestrian":"The pedestrian, a male in his 30s with a height of 170 cm, was dressed in a brown jacket and navy blue slacks. He stood still diagonally to the left in front of the vehicle on a clear and bright day. The pedestrian's body was facing the same direction as the vehicle, both of which were on a two-way residential road with a dry asphalt surface. The road had usual traffic volume and was relatively level. There was no sidewalk or roadside strip on both sides. Despite the usual traffic, the pedestrian appeared calm and stationary as he faced the front.",
         "caption_vehicle":"The vehicle is positioned behind and to the right of the pedestrian. It is near to the pedestrian and the pedestrian is not visible within the vehicle's field of view. The vehicle is moving straight ahead at a speed of 20 km/h. The male pedestrian in his 30s stands at a height of 170 cm. He is wearing a brown jacket and navy blue slacks. The weather is clear with bright brightness. The road surface is dry and level, with asphalt as the road surface type. The traffic volume is usual on this two-way residential road with no sidewalk on both sides or roadside strips.",
         "start_time":"33.527",
         "end_time":"34.844"
      },
      {
         "labels":[
            "0"
         ],
         "caption_pedestrian":"The pedestrian is a male in his 30s with a height of 170 cm. He is wearing a brown jacket and navy blue slacks. It is a clear and bright day with dry road surface conditions. The road is a residential road with two-way traffic and no sidewalks on both sides. As the pedestrian stands still, he is positioned diagonally to the left in front of the vehicle. The orientation of his body is the same direction as the vehicle, and he is relatively far from it. The pedestrian's direction of travel is in front of the vehicle. The surroundings reveal an inclined road surface made of asphalt. The traffic volume is usual, and there is no roadside strip on either side. This concise description captures the key details of the pedestrian's appearance, their action, and the environmental conditions in a clear and easily understandable manner.",
         "caption_vehicle":"The vehicle is positioned behind and to the right of the pedestrian, with a relative distance that is far. However, the vehicle's field of view does not allow them to see the pedestrian. The vehicle is going straight ahead at a speed of 5 km/h. In the environment, the pedestrian is a male in his 30s, approximately 170 cm tall, wearing a brown jacket and navy blue slacks. The weather is clear and the brightness level is bright. The road surface conditions are dry and level, with asphalt as the road surface type. The traffic volume is usual on this residential road, which has two-way traffic and sidewalks on only one side. There are no roadside strips on either side of the road.",
         "start_time":"32.210",
         "end_time":"33.527"
      }
   ],
   "20230728_16_SY19_T1":[
      {
         "labels":[
            "1"
         ],
         "caption_pedestrian":"The pedestrian, a man in his 30s with a height of 170 cm, was standing still on the road, diagonally to the left in front of the vehicle. He was wearing a black T-shirt and black slacks, giving a bright appearance against the clear weather. The road surface was dry and level. The pedestrian was unaware of the vehicle approaching him. His line of sight was in front, in the direction of travel, aligning with the vehicle's movement. He seemed to be facing the same direction as the vehicle, indicating his body orientation. The relative distance between the pedestrian and the vehicle was near. Despite the surroundings, the pedestrian remained stationary in his actions.",
         "caption_vehicle":"The vehicle is currently positioned behind and to the right of a pedestrian. It is relatively close to the pedestrian, and the driver has a clear view of them. The vehicle is proceeding straight ahead without any deviations. The environment surrounding the event consists of a male pedestrian in his 30s. He is approximately 170 cm tall and is wearing a black T-shirt on his upper body and black slacks on his lower body. The overall brightness of the environment is bright, and the weather is clear. The road surface is dry and level, making it easier for the vehicle to maintain stability while driving.",
         "start_time":"33.555",
         "end_time":"34.411"
      },
      {
         "labels":[
            "0"
         ],
         "caption_pedestrian":"The pedestrian, a male in his 30s, stood near the vehicle. His body was oriented in the same direction as the vehicle, and he was positioned diagonally to the left in front of it. His line of sight was focused in front, in the direction of travel. Closely watching his surroundings, he seemed unaware of the vehicle's presence. His general action was to continue going straight ahead at a slow speed. The environment conditions around him were bright, with clear weather and a dry, level road surface. He was wearing a black T-shirt and black slacks and stood at a height of 170 cm. Overall, the pedestrian appeared to be calm and unaware of any potential danger from the vehicle as he continued on his path ahead.",
         "caption_vehicle":"The vehicle was situated behind and to the right of a pedestrian, at a relatively close distance. The pedestrian was clearly visible within the vehicle's field of view. The vehicle was about to initiate movement at a speed of 5 km/h. Meanwhile, the environment condition indicated that the pedestrian was a male in his 30s, with a height of 170 cm. He was wearing a black T-shirt on the upper body and black slacks on the lower body. The brightness of the surroundings was bright, with clear weather and a dry, level road surface. These were the circumstances surrounding the event involving the vehicle, presenting a concise and objective depiction of the situation.",
         "start_time":"32.582",
         "end_time":"33.555"
      },
      {
         "labels":[
            "2"
         ],
         "caption_pedestrian":"The pedestrian, a male in his 30s wearing a black T-shirt and black slacks, was walking unaware of the vehicle approaching him. He was oriented in the same direction as the vehicle, positioned diagonally to the left in front of it. His line of sight was focused in front of him, aligned with the direction of travel. Going straight ahead, he seemed to be unaware of the vehicle and its presence. The environment conditions were bright with clear weather and a dry road surface. The road was level and the pedestrian seemed to be comfortable in his surroundings. Overall, the pedestrian appeared to be engaged in his own activities, displaying a calm demeanor as he continued walking straight ahead without encountering any hindrances.",
         "caption_vehicle":"The vehicle is located behind and to the right of the pedestrian, at a close distance. The pedestrian is visible within the vehicle's field of view. The vehicle is moving straight ahead at a speed of 5 km/h. In terms of the environment, the pedestrian is a male in his 30s with a height of 170 cm. He is wearing a black T-shirt and black slacks. The brightness is bright, and the weather is clear. The road surface conditions are dry and level.",
         "start_time":"34.419",
         "end_time":"35.313"
      },
      {
         "labels":[
            "4"
         ],
         "caption_pedestrian":"The pedestrian, a man in his 30s, was walking in the same direction as the vehicle. He was positioned diagonally to the left in front of the vehicle, at a close distance. His line of sight was directed in front, following the direction of travel. The pedestrian was closely watching his surroundings, unaware of the vehicle's presence. Suddenly, a collision occurred between the pedestrian and the vehicle. The man's attire consisted of a black T-shirt and black slacks, with a height of 170 cm. The environment conditions were bright and clear, with dry road surface conditions and a level incline. Despite the favorable weather and road conditions, the accident took place due to the pedestrian's lack of awareness of the vehicle's presence.",
         "caption_vehicle":"The vehicle is positioned behind and to the right of the pedestrian, and they are close to each other. The pedestrian is clearly visible within the vehicle's field of view. The vehicle is going straight ahead, but its speed is 0 km/h, indicating that it is stationary. The male pedestrian is in his 30s and has a height of approximately 170 cm. He is wearing a black T-shirt and black slacks. The environmental conditions are bright, with clear weather and a dry road surface that is level.",
         "start_time":"40.305",
         "end_time":"43.313"
      },
      {
         "labels":[
            "3"
         ],
         "caption_pedestrian":"The pedestrian is a male in his 30s, approximately 170 cm tall. He is wearing a black T-shirt and black slacks. The weather is clear, and the road surface is dry and level. The brightness of the environment is high. As the pedestrian walks slowly, his body is oriented in the same direction as the vehicle. He is positioned diagonally to the left in front of the vehicle, and his relative distance from the vehicle is close. His line of sight is in front, in the direction of travel, and he is closely watching his surroundings. However, he is unaware of the vehicle approaching. The pedestrian's general action is to continue going straight ahead.",
         "caption_vehicle":"The vehicle is positioned behind and to the right of the pedestrian, indicating that it is following behind at a close distance. The pedestrian is visible within the vehicle's field of view. The vehicle is moving straight ahead with a speed of 5 km/h. In the environment, the person crossing the road is a male in his 30s with a height of 170 cm. He is wearing a black T-shirt on the upper body and black slacks on the lower body. The brightness of the surroundings is bright, with clear weather conditions and a dry road surface that is level. Based on this information, it can be inferred that the vehicle is driving in a cautious manner as it maintains a close distance behind the pedestrian. The pedestrian's visibility suggests that the driver is aware of their presence. The environment conditions, including the brightness and weather, provide optimal visibility for both the vehicle and the pedestrian. The dry road surface and level incline further contribute to safe driving conditions. Overall, the event portrays a typical scenario where the vehicle is following a pedestrian while adhering to appropriate driving speed and conditions.",
         "start_time":"35.321",
         "end_time":"40.330"
      }
   ]
}

pred = {
    "20230707_8_SN46_T1": [
        {
            "labels": [
                "4"
            ],
            "caption_pedestrian": "The person stands still on the right side behind the vehicle. His line of sight focuses on the vehicle itself. His awareness of the vehicle is not complete. In his 30s, the man is of male gender and stands at a height of 170 cm. He is wearing a brown jacket on his upper body and navy blue slacks on his lower body. The pedestrian's general action is standing still, but there is also a special action of rushing out, albeit slowly.",
            "caption_vehicle": "The vehicle was positioned diagonally to the left in front of the pedestrian, and the relative distance between the two was close. The vehicle's field of view indicated that the pedestrian was not visible to the driver. Unfortunately, the vehicle collided with the pedestrian while moving at a speed of 20 km/h. Meanwhile, the environment conditions revealed that the pedestrian was a male in his 30s, standing at a height of 170 cm. He was wearing a brown jacket and navy blue slacks. The weather was clear and bright, and the road surface was dry and level. The road itself was a residential one with two-way traffic and no sidewalks present on both sides. Additionally, the road surface was made of asphalt. The traffic volume was usual, and there were no roadside strips on both sides.",
            "start_time": "39.395",
            "end_time": "44.663"
        },
        {
            "labels": [
                "3"
            ],
            "caption_pedestrian": "The weather is clear and bright. The road surface conditions are dry and the road is level asphalt. The traffic volume is usual, and it is a residential road with two-way traffic and no sidewalks on both sides or roadside strips. The pedestrian's body is positioned perpendicular to the vehicle and to the right side of it. He is close to the vehicle and notices its presence. The pedestrian is standing still but suddenly rushes out. He moves slowly towards the right direction.",
            "caption_vehicle": "The vehicle is positioned on the left side of the pedestrian and is fairly close to them. The pedestrian is within the vehicle's field of view, so they are visible to the driver. The vehicle's current action is to continue moving straight ahead, and it is traveling at a speed of 20 km/h. The environment conditions surrounding the event involve a male pedestrian in his 30s with a height of 170 cm. He is wearing a brown jacket on the upper body and navy blue slacks on the lower body. The weather is clear, with bright visibility. The road surface is dry and level, made of asphalt. The traffic volume is usual, and the road where the event is taking place is a residential road with two-way traffic. There is no sidewalk on both sides of the road, and there is no roadside strip present.Overall, the vehicle is in close proximity to the pedestrian and has clear visibility of them. The environment conditions, such as the weather and road surface, are optimal for driving.",
            "start_time": "36.162",
            "end_time": "39.395"
        },
        {
            "labels": [
                "2"
            ],
            "caption_pedestrian": "The pedestrian, a male in his 30s, stood near the vehicle with his body perpendicular and to the right of it. He wore a brown jacket on his upper body and navy blue slacks on his lower body. The weather was clear, and the brightness of the surroundings was bright. The road surface was dry, and the road was level, with asphalt. The residential road had two-way traffic with no sidewalk on both sides, nor a roadside strip. The pedestrian seemed unaware of the vehicle and was about to rush out in front of it. The event took place in usual traffic conditions.",
            "caption_vehicle": "The vehicle is positioned on the right side of the pedestrian at a near distance. The pedestrian is visible within the vehicle's field of view. The vehicle is going straight ahead at a speed of 20 km/h. The environment conditions include a male in his 30s with a height of 170 cm. He is wearing a brown jacket and navy blue slacks. The weather is clear, with a bright brightness level. The road surface conditions are dry and level, with asphalt as the road surface type. The traffic volume is usual on this residential road with two-way traffic. There is no sidewalk on both sides, and there is no roadside strip on both sides.",
            "start_time": "34.725",
            "end_time": "36.162"
        },
        {
            "labels": [
                "1"
            ],
            "caption_pedestrian": "The pedestrian, a male in his 30s with a height of 170 cm, was dressed in a brown jacket and navy blue slacks. He stood still diagonally to the left in front of the vehicle on a clear and bright day. The pedestrian's body was facing the same direction as the vehicle, both of which were on a two-way residential road with a dry asphalt surface. The road had usual traffic volume and was relatively level. There was no sidewalk or roadside strip on both sides. Despite the usual traffic, the pedestrian appeared calm and stationary as he faced the front.",
            "caption_vehicle": "The vehicle is positioned behind and to the right of the pedestrian. It is near to the pedestrian and the pedestrian is not visible within the vehicle's field of view. The vehicle is moving straight ahead at a speed of 20 km/h. The male pedestrian in his 30s stands at a height of 170 cm. He is wearing a brown jacket and navy blue slacks. The weather is clear with bright brightness. The road surface is dry and level, with asphalt as the road surface type. The traffic volume is usual on this two-way residential road with no sidewalk on both sides or roadside strips.",
            "start_time": "33.527",
            "end_time": "34.844"
        },
        {
            "labels": [
                "0"
            ],
            "caption_pedestrian": "The pedestrian is a male in his 30s with a height of 170 cm. He is wearing a brown jacket and navy blue slacks. It is a clear and bright day with dry road surface conditions. The road is a residential road with two-way traffic and no sidewalks on both sides. As the pedestrian stands still, he is positioned diagonally to the left in front of the vehicle. The orientation of his body is the same direction as the vehicle, and he is relatively far from it. The pedestrian's direction of travel is in front of the vehicle. The surroundings reveal an inclined road surface made of asphalt. The traffic volume is usual, and there is no roadside strip on either side. This concise description captures the key details of the pedestrian's appearance, their action, and the environmental conditions in a clear and easily understandable manner.",
            "caption_vehicle": "The vehicle is positioned behind and to the right of the pedestrian, with a relative distance that is far. However, the vehicle's field of view does not allow them to see the pedestrian. The vehicle is going straight ahead at a speed of 5 km/h. In the environment, the pedestrian is a male in his 30s, approximately 170 cm tall, wearing a brown jacket and navy blue slacks. The weather is clear and the brightness level is bright. The road surface conditions are dry and level, with asphalt as the road surface type. The traffic volume is usual on this residential road, which has two-way traffic and sidewalks on only one side. There are no roadside strips on either side of the road.",
            "start_time": "32.210",
            "end_time": "33.527"
        }
    ],

    "20230728_16_SY19_T1": [
        {
            "labels": [
                "1"
            ],
            "caption_pedestrian": "The pedestrian, a man in his 30s with a height of 170 cm, was standing still on the road, diagonally to the left in front of the vehicle. He was wearing a black T-shirt and black slacks, giving a bright appearance against the clear weather. The road surface was dry and level. The pedestrian was unaware of the vehicle approaching him. His line of sight was in front, in the direction of travel, aligning with the vehicle's movement. He seemed to be facing the same direction as the vehicle, indicating his body orientation. The relative distance between the pedestrian and the vehicle was near. Despite the surroundings, the pedestrian remained stationary in his actions.",
            "caption_vehicle": "The vehicle is currently positioned behind and to the right of a pedestrian. It is relatively close to the pedestrian, and the driver has a clear view of them. The vehicle is proceeding straight ahead without any deviations. The environment surrounding the event consists of a male pedestrian in his 30s. He is approximately 170 cm tall and is wearing a black T-shirt on his upper body and black slacks on his lower body. The overall brightness of the environment is bright, and the weather is clear. The road surface is dry and level, making it easier for the vehicle to maintain stability while driving.",
            "start_time": "33.555",
            "end_time": "34.411"
        },
        {
            "labels": [
                "0"
            ],
            "caption_pedestrian": "The pedestrian, a male in his 30s, stood near the vehicle. His body was oriented in the same direction as the vehicle, and he was positioned diagonally to the left in front of it. His line of sight was focused in front, in the direction of travel. Closely watching his surroundings, he seemed unaware of the vehicle's presence. His general action was to continue going straight ahead at a slow speed. The environment conditions around him were bright, with clear weather and a dry, level road surface. He was wearing a black T-shirt and black slacks and stood at a height of 170 cm. Overall, the pedestrian appeared to be calm and unaware of any potential danger from the vehicle as he continued on his path ahead.",
            "caption_vehicle": "The vehicle was situated behind and to the right of a pedestrian, at a relatively close distance. The pedestrian was clearly visible within the vehicle's field of view. The vehicle was about to initiate movement at a speed of 5 km/h. Meanwhile, the environment condition indicated that the pedestrian was a male in his 30s, with a height of 170 cm. He was wearing a black T-shirt on the upper body and black slacks on the lower body. The brightness of the surroundings was bright, with clear weather and a dry, level road surface. These were the circumstances surrounding the event involving the vehicle, presenting a concise and objective depiction of the situation.",
            "start_time": "32.582",
            "end_time": "33.555"
        },
        {
            "labels": [
                "2"
            ],
            "caption_pedestrian": "The pedestrian, a male in his 30s wearing a black T-shirt and black slacks, was walking unaware of the vehicle approaching him. He was oriented in the same direction as the vehicle, positioned diagonally to the left in front of it. His line of sight was focused in front of him, aligned with the direction of travel. Going straight ahead, he seemed to be unaware of the vehicle and its presence. The environment conditions were bright with clear weather and a dry road surface. The road was level and the pedestrian seemed to be comfortable in his surroundings. Overall, the pedestrian appeared to be engaged in his own activities, displaying a calm demeanor as he continued walking straight ahead without encountering any hindrances.",
            "caption_vehicle": "The vehicle is located behind and to the right of the pedestrian, at a close distance. The pedestrian is visible within the vehicle's field of view. The vehicle is moving straight ahead at a speed of 5 km/h. In terms of the environment, the pedestrian is a male in his 30s with a height of 170 cm. He is wearing a black T-shirt and black slacks. The brightness is bright, and the weather is clear. The road surface conditions are dry and level.",
            "start_time": "34.419",
            "end_time": "35.313"
        },
        {
            "labels": [
                "4"
            ],
            "caption_pedestrian": "The pedestrian, a man in his 30s, was walking in the same direction as the vehicle. He was positioned diagonally to the left in front of the vehicle, at a close distance. His line of sight was directed in front, following the direction of travel. The pedestrian was closely watching his surroundings, unaware of the vehicle's presence. Suddenly, a collision occurred between the pedestrian and the vehicle. The man's attire consisted of a black T-shirt and black slacks, with a height of 170 cm. The environment conditions were bright and clear, with dry road surface conditions and a level incline. Despite the favorable weather and road conditions, the accident took place due to the pedestrian's lack of awareness of the vehicle's presence.",
            "caption_vehicle": "The vehicle is positioned behind and to the right of the pedestrian, and they are close to each other. The pedestrian is clearly visible within the vehicle's field of view. The vehicle is going straight ahead, but its speed is 0 km/h, indicating that it is stationary. The male pedestrian is in his 30s and has a height of approximately 170 cm. He is wearing a black T-shirt and black slacks. The environmental conditions are bright, with clear weather and a dry road surface that is level.",
            "start_time": "40.305",
            "end_time": "43.313"
        },
        {
            "labels": [
                "3"
            ],
            "caption_pedestrian": "The pedestrian is a male in his 30s, approximately 170 cm tall. He is wearing a black T-shirt and black slacks. The weather is clear, and the road surface is dry and level. The brightness of the environment is high. As the pedestrian walks slowly, his body is oriented in the same direction as the vehicle. He is positioned diagonally to the left in front of the vehicle, and his relative distance from the vehicle is close. His line of sight is in front, in the direction of travel, and he is closely watching his surroundings. However, he is unaware of the vehicle approaching. The pedestrian's general action is to continue going straight ahead.",
            "caption_vehicle": "The vehicle is positioned behind and to the right of the pedestrian, indicating that it is following behind at a close distance. The pedestrian is visible within the vehicle's field of view. The vehicle is moving straight ahead with a speed of 5 km/h. In the environment, the person crossing the road is a male in his 30s with a height of 170 cm. He is wearing a black T-shirt on the upper body and black slacks on the lower body. The brightness of the surroundings is bright, with clear weather conditions and a dry road surface that is level. Based on this information, it can be inferred that the vehicle is driving in a cautious manner as it maintains a close distance behind the pedestrian. The pedestrian's visibility suggests that the driver is aware of their presence. The environment conditions, including the brightness and weather, provide optimal visibility for both the vehicle and the pedestrian. The dry road surface and level incline further contribute to safe driving conditions. Overall, the event portrays a typical scenario where the vehicle is following a pedestrian while adhering to appropriate driving speed and conditions.",
            "start_time": "35.321",
            "end_time": "40.330"
        }
    ]
}


def probe_metrics():
  print("Checking evaluation pipeline...")
  gt_sent = convert_dict_to_desc_list(gt)
  pred_sent = convert_dict_to_desc_list(pred)
  return batch_evaluate(gt_sent, pred_sent)
