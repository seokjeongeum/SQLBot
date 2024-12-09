from typing import *

from table2text.llama import LLMAPI


class Table2TextModel:
    def __init__(
        self,
        use_text_generation: bool = True,
        api_addr: Optional[str] = None,
        model_name: Optional[str] = None,
        precision: Optional[int] = None,
    ):
        super().__init__()
        self.use_text_generation = use_text_generation
        self.api_addr = api_addr
        self.model_name = model_name
        self.precision = precision
        self.__post_init__()

    def __post_init__(self) -> None:
        self.llm = LLMAPI(
            use_text_generation=self.use_text_generation,
            api_addr=self.api_addr,
            model_name=self.model_name,
            precision=self.precision,
            override_cache=True,
        )

    @property
    def instruction(self) -> str:
        return "Summarize the given table into one sentence. Do not include extra information."

    @property
    def few_shot_examples(self) -> List[str]:
        return [
            {
                "table": [
                    {
                        "singer_id": "1",
                        "name": "Joe Sharp",
                        "country": "Netherlands",
                        "song_name": "You",
                        "song_release_year": "1992",
                        "age": "52",
                        "is_male": "F",
                    },
                    {
                        "singer_id": "2",
                        "name": "Timbaland",
                        "country": "United States",
                        "song_name": "Dangerous",
                        "song_release_year": "2008",
                        "age": "32",
                        "is_male": "T",
                    },
                    {
                        "singer_id": "3",
                        "name": "Justin Brown",
                        "country": "Frane",
                        "song_name": "Hey Oh",
                        "song_release_year": "2013",
                        "age": "20",
                        "is_male": "T",
                    },
                    {
                        "singer_id": "4",
                        "name": "Rose White",
                        "country": "Frane",
                        "song_name": "Sun",
                        "song_release_year": "2003",
                        "age": "41",
                        "is_male": "F",
                    },
                    {
                        "singer_id": "5",
                        "name": "John Nizinik",
                        "country": "Frane",
                        "song_name": "Gentleman",
                        "song_release_year": "2014",
                        "age": "43",
                        "is_male": "T",
                    },
                ],
                "summary": "The table summarizes data on five singers from the Netherlands, United States, and France, detailing their names, song titles, release years, ages, and genders, with songs ranging from 1992 to 2014 and ages from 20 to 52.",
            },
            {
                "table": [
                    {
                        "stuid": "1001",
                        "Iname": "Smith",
                        "fname": "Linda",
                        "age": "18",
                        "sex": "F",
                        "major": "600",
                        "advisor": "1121",
                        "city_code": "BAL",
                    },
                    {
                        "stuid": "1002",
                        "Iname": "Kim",
                        "fname": "Tracy",
                        "age": "19",
                        "sex": "F",
                        "major": "600",
                        "advisor": "7712",
                        "city_code": "HKG",
                    },
                    {
                        "stuid": "1003",
                        "Iname": "Jones",
                        "fname": "Shiela",
                        "age": "21",
                        "sex": "F",
                        "major": "600",
                        "advisor": "7792",
                        "city_code": "WAS",
                    },
                ],
                "summary": "The table presents details of three female students aged 18 to 21, named Linda Smith, Tracy Kim, and Shiela Jones, all majoring in the same field (600), with different advisors and hailing from cities BAL, HKG, and WAS respectively.",
            },
            {
                "table": [
                    {
                        "stadium_id": "5",
                        "location": "Stirling Albion",
                        "name": "Forthbank Stadium",
                        "capacity": "3808",
                        "highest": "1125",
                        "lowest": "404",
                        "average": "642",
                    },
                ],
                "summary": "The table provides information on Forthbank Stadium, the home of Stirling Albion, with a capacity of 3,808 and attendance statistics showing a highest of 1,125, lowest of 404, and an average of 642.",
            },
        ]

    @property
    def table_prefix(self) -> str:
        return "Table: "

    @property
    def summary_prefix(self) -> str:
        return "Summary: "

    def table_to_string(self, table: List[Dict]) -> str:
        # Extract column names
        column_names = list(table[0].keys())
        # Extract row values
        row_values = [list(row.values()) for row in table]
        # Convert to string
        table_in_string = "Column Names: " + ", ".join(column_names) + ".\n"
        for idx, row in enumerate(row_values, start=1):
            # If row is not string, convert to string
            row = [str(item) for item in row]
            table_in_string += f"Row {idx}: " + "(" + ", ".join(row) + ")\n"
        return table_in_string

    def format_input(
        self,
        table_in_dict: List[Dict],
    ) -> str:
        # Convert table to string
        table_in_string = self.table_to_string(table=table_in_dict)

        user_prompt: str = ""
        # Append few-shot examples
        for example in self.few_shot_examples:
            # Append table
            example_prompt = (
                self.table_prefix + self.table_to_string(example["table"]) + "\n"
            )
            # Append summary
            # Append answer
            example_prompt += self.summary_prefix + example["summary"] + "\n"
            # Append example
            user_prompt += example_prompt + "\n"

        # Append current question
        user_prompt += self.table_prefix + table_in_string + "\n"

        return user_prompt

    def parse_response(self, response: str) -> str:
        if self.summary_prefix in response:
            return response.split(self.summary_prefix)[1]
        return response

    def infer(
        self,
        table_in_dict: List[Dict],
    ) -> str:
        # Prepare instruction prompts
        instruction_prompt = self.instruction

        # Prepare user prompts
        user_prompt = self.format_input(table_in_dict=table_in_dict)

        # Generate
        response = self.llm.generate(
            instruction_prompt=instruction_prompt,
            user_prompt=user_prompt,
            output_prefix=self.summary_prefix,
        ).strip()

        summary = self.parse_response(response=response)
        return summary

    def batch_infer(
        self,
        list_of_table_in_dict: List[List[Dict]],
    ) -> List[str]:
        return [
            self.infer(table_in_dict=table_in_dict)
            for table_in_dict in list_of_table_in_dict
        ]


if __name__ == "__main__":
    model = Table2TextModel()
    table_in_dict = [
        {
            "concert_id": "1",
            "concert_name": "Auditions",
            "theme": "Free choice",
            "stadium_id": "1",
            "year": "2014",
        },
        {
            "concert_id": "3",
            "concert_name": "Home Visits",
            "theme": "Bleeding Love",
            "stadium_id": "2",
            "year": "2015",
        },
        {
            "concert_id": "6",
            "concert_name": "Week 2",
            "theme": "Party All Night",
            "stadium_id": "7",
            "year": "2015",
        },
    ]
    result = model.infer(table_in_dict=table_in_dict)
    print(result)
