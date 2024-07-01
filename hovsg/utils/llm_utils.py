from functools import lru_cache
import os
from typing import Dict, List, Tuple, Union

import openai


def infer_floor_id_from_query(floor_ids: List[int], query: str) -> int:
    """return the floor id from the floor_ids_list that match with the query

    Args:
        floor_ids (List[int]): a list starting from 1 to highest floor level
        query (str): a text description of the floor level number

    Returns:
        int: the target floor number (starting from 1)
    """
    floor_ids_str = [str(i) for i in floor_ids]
    floor_ids_str = ", ".join(floor_ids_str)

    openai_key = os.environ["OPENAI_KEY"]
    openai.api_key = openai_key
    question = f"""
You are a floor detector. You can infer the floor number based on a query.
The query is: {query}.
The floor number list is: {floor_ids_str}.
Please answer the floor number in one integer.
    """
    print(question)
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=question,
        max_tokens=64,
        temperature=0.0,
        stop=None,
    )
    result = response["choices"][0]["text"]
    try:
        result = int(result)
    except:
        print(f"The return answer is not an integer. The answer is: {result}")
        assert False
    return result


def infer_room_type_from_object_list_chat(
    object_list: List[str], default_room_type: List[str] = None
) -> str:
    """generate a room type based on a list of objects contained in the room with chat

    Args:
        object_list (List[str]): a list of object names contained in the room
        default_room_type (List[str] = None): the inferred room type should be from this list

    Returns:
        str: a text describing the room type
    """
    openai_key = os.environ["OPENAI_KEY"]
    openai.api_key = openai_key
    client = openai.OpenAI(api_key=openai_key)
    room_types = ""
    if default_room_type is not None:
        room_types = ", ".join(default_room_type)
        room_types = (
            "Please pick the most matching room type from the following list: "
            + room_types
            + "."
        )

    objects = ", ".join(object_list)
    print(f"Objects list: {objects}")
    print(f"Room types: {room_types}")

    question = f"""
    """
    print(question)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a room type detector. You can infer a room type based on a list of objects.",
            },
            {
                "role": "user",
                "content": f"The list of objects contained in this room are: bed, wardrobe, chair, sofa. What is the room type? Please just answer the room name.",
            },
            {
                "role": "assistant",
                "content": f"bedroom",
            },
            {
                "role": "user",
                "content": f"The list of objects contained in this room are: tv, table, chair, sofa. Please pick the most matching room type from the following list: living room, bedroom, bathroom, kitchen. What is the room type? Please just answer the room name.",
            },
            {
                "role": "assistant",
                "content": f"living room",
            },
            {
                "role": "user",
                "content": f"The list of objects contained in this room are: {objects}. {room_types} What is the room type? Please just answer the room name.",
            },
        ],
    )
    print(response)
    result = response.choices[0].message.content
    print("The room type is: ", result)
    return result


class Conversation:
    def __init__(
        self, messages: List[dict], include_env_messages: bool = False
    ) -> None:
        """An interface to OPENAI chat API

        Args:
            messages (List[dict]): The list of messages to be sent to the chat API
            include_env_messages (bool, optional): Boolean controlling if environment message is sent. Defaults to False.
        """
        self._messages = messages
        self._include_env_messages = include_env_messages

    def add_message(self, message: dict):
        self._messages.append(message)

    @property
    def messages(self):
        if self._include_env_messages:
            return self._messages
        else:
            return [
                m
                for m in self._messages
                if m["role"].lower() not in ["env", "environment"]
            ]

    @property
    def messages_including_env(self):
        return self._messages


@lru_cache(maxsize=None)
def send_query_cached(client, messages: list, model: str, temperature: float):
    assert (
        temperature == 0.0
    ), "Caching only works for temperature=0.0, as eitherwise we want to get different responses back"
    messages = [dict(m) for m in messages]
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )


def send_query(client, messages: list, model: str, temperature: float):
    # if temperature == 0.0:
    #     hashable_messages = tuple(tuple(m.items()) for m in messages)
    #     return send_query_cached(client, messages=hashable_messages, model=model, temperature=temperature)
    # else:
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )


def parse_hier_query(params, instruction: str) -> Tuple[str, str, str]:
    """
    Parse long language query into a list of short queries at floor, room, and object level
    Example: "mirror in region bathroom on floor 0" -> ("floor 0", "bathroom", "mirror")
    """
    openai_key = os.environ["OPENAI_KEY"]
    openai.api_key = openai_key
    client = openai.OpenAI(api_key=openai_key)

    # Depending on the query spec, parse the query differently:
    if set(params.main.long_query.spec) == {"obj", "room", "floor"}:
        system_prompt = "You are a query parser. You have to parse a sentence into a floor, a room and an object."
        prompt = f"Please parse the following: {instruction}"
        prompt += "Output Response Format: Comma-separated list of these three things such as [floor 2, living room, couch]"
    elif set(params.main.long_query.spec) == {"obj", "room"}:
        system_prompt = "You are a query parser. You have to parse a sentence into a room and an object."
        prompt = f"Please parse the following: {instruction}"
        prompt += "Output Response Format: Comma-separated list of these two things such as [living room, couch]"
    elif set(params.main.long_query.spec) == {"obj", "floor"}:
        system_prompt = "You are a query parser. You have to parse a sentence into a floor and an object."
        prompt = f"Please parse the following: {instruction}"
        prompt += "Output Response Format: Comma-separated list of these two things such as [floor 2, couch]"
    elif set(params.main.long_query.spec) == {"obj"}:
        # return directly and not use the LLM for parsing
        print("floor, room, object:", None, None, instruction)
        return [None, None, instruction.strip()]

    conversation = Conversation(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    )

    response = send_query(
        client, messages=conversation.messages, model="gpt-3.5-turbo", temperature=0.0
    )
    result = response.choices[0].message.content.strip().rstrip("]").lstrip("[")

    if set(params.main.long_query.spec) == {"floor", "room", "obj"}:
        print("floor, room, object:", result)
        return [x.strip() for x in result.split(",")]
    elif set(params.main.long_query.spec) == {"room", "obj"}:
        print("floor, room, object:", None, result)
        return [None, result.split(",")[0].strip(), result.split(",")[1].strip()]
    elif set(params.main.long_query.spec) == {"floor", "obj"}:
        print("floor, room, object:", result.split(",")[0], None, result.split(",")[1])
        return [result.split(",")[0].strip(), None, result.split(",")[1].strip()]


def parse_floor_room_object_gpt35(instruction: str) -> Tuple[str, str, str]:
    """
    Parse long language query into a list of short queries at floor, room, and object level
    Example: "mirror in region bathroom on floor 0" -> ("floor 0", "bathroom", "mirror")
    """

    openai_key = os.environ["OPENAI_KEY"]
    openai.api_key = openai_key
    client = openai.OpenAI(api_key=openai_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a hierarchical concept parser. You need to parse a description of an object into floor, region and object.",
            },
            {
                "role": "user",
                "content": "chair in region living room on the 0 floor",
            },
            {"role": "assistant", "content": "[floor 0,living room,chair]"},
            {
                "role": "user",
                "content": "floor in living room on floor 0",
            },
            {"role": "assistant", "content": "[floor 0,living room,floor]"},
            {
                "role": "user",
                "content": "table in kitchen on floor 3",
            },
            {"role": "assistant", "content": "[floor 3,kitchen,table]"},
            {
                "role": "user",
                "content": "cabinet in region bedroom on floor 1",
            },
            {"role": "assistant", "content": "[floor 1,bedroom,cabinet]"},
            {
                "role": "user",
                "content": "bedroom on floor 1",
            },
            {"role": "assistant", "content": "[floor 1,bedroom,]"},
            {
                "role": "user",
                "content": "bed",
            },
            {"role": "assistant", "content": "[,,bed]"},
            {
                "role": "user",
                "content": "bedroom",
            },
            {"role": "assistant", "content": "[,bedroom,]"},
            {
                "role": "user",
                "content": "I want to go to bed, where should I go?",
            },
            {"role": "assistant", "content": "[,bedroom,]"},
            {
                "role": "user",
                "content": "I want to go for something to eat upstairs. I am currently at floor 0, where should I go?",
            },
            {"role": "assistant", "content": "[floor 1,dinning,]"},
            {
                "role": "user",
                "content": f"{instruction}",
            },
        ],
    )
    print(response.choices[0].message.content)
    result = response.choices[0].message.content.strip().rstrip("]").lstrip("[")
    print("floor, room, object:", result)
    decomposition = [x.strip() for x in result.split(",")]
    assert len(decomposition) == 3 and (
        decomposition[0] != "" or decomposition[1] != "" or decomposition[2] != ""
    )
    return decomposition


def main():
    # result = parse_floor_room_object_gpt35("picture in region bedroom on floor 1")
    # object_list = ["sink", "soap", "towel", "hair dryer"]
    object_list = [
        "carpet",
        "counter",
        "baseball bat",
        "metal",
        "carpet",
        "banner",
        "blanket",
        "curtain",
        "dining table",
        "shelf",
        "cupboard",
        "curtain",
        "road",
        "banner",
        "banner",
        "oven",
        "carpet",
        "metal",
        "skateboard",
        "mirror",
        "bowl",
        "shelf",
        "mud",
        "cupboard",
        "window",
        "cupboard",
        "paper",
        "banner",
        "waterdrops",
        "waterdrops",
        "umbrella",
        "curtain",
        "refrigerator",
        "banner",
        "solid-other",
        "waterdrops",
        "clothes",
        "solid-other",
        "wood",
        "paper",
        "solid-other",
        "solid-other",
        "metal",
        "solid-other",
        "waterdrops",
        "bottle",
        "orange",
        "hat",
        "banner",
        "couch",
        "wood",
        "wood",
        "metal",
        "paper",
        "wood",
        "orange",
        "banner",
        "tv",
        "tv",
        "cupboard",
        "banner",
        "oven",
        "furniture-other",
        "cardboard",
        "metal",
        "banner",
        "hat",
        "curtain",
        "orange",
        "stone",
        "fog",
        "sink",
        "metal",
        "hat",
        "metal",
        "metal",
        "leaves",
    ]
    # default_list = ["guest room", "kitchen", "bathroom", "bedroom"]
    # result = infer_room_type_from_object_list(object_list, default_list)
    # result = infer_room_type_from_object_list_chat(object_list)
    # result = infer_floor_id_from_query([0, 1, 2, 3, 4], "floor 0")
    while True:
        instruction = input("Enter instruction: ")
        result = parse_floor_room_object_gpt35(instruction)
        print(result)
    # result = parse_floor_room_object_gpt35(
    #     "I want to cook something downstairs. I am currently at floor 1, where should I go?"
    # )
    # result = parse_floor_room_object_gpt35("cabinet")
    # print(result)


if __name__ == "__main__":
    main()
