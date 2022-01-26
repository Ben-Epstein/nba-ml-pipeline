import requests
from zenml.steps import step


@step
def discord_alert(
    input: dict,
) -> bool:
    """Send a message to the discord channel to report drift."""
    drift = input['data_drift']['data']['metrics']['dataset_drift']
    url = "https://discord.com/api/webhooks/935835443826659339/Q32jTwmqcGJAUr-" \
          "r_J3ouO-zkNQPchJHqTuwJ7dK4wiFzawT2Gu97f6ACt58UKFCxEO9"
    data = {
        "content": "Drift Detected!" if drift else "Drift Not Detected!",
        "username": "Drift Bot"
    }
    result = requests.post(url, json=data)

    try:
        result.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(err)
    else:
        print("Payload delivered successfully, code {}.".format(result.status_code))
    print("Drift detected" if drift else "No drift detected")
    return drift
