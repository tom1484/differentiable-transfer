def args_to_commands(**kwargs) -> str:
    commands = []
    for k, v in kwargs.items():
        if type(v) == bool:
            if v:
                commands.append(f"--{k.replace('_', '-')}")
            else:
                commands.append(f"--no-{k.replace('_', '-')}")
        else:
            commands.append(f"--{k.replace('_', '-')}={v}")

    return commands