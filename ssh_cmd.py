#!/usr/bin/env python3
"""
Reusable SSH helper – runs commands on the remote server without manual password input.
Usage:
    python ssh_cmd.py "command1; command2; ..."
    python ssh_cmd.py -f local_script.sh          # upload & run a script
    python ssh_cmd.py --get remote_path local_path # download file
    python ssh_cmd.py --put local_path remote_path # upload file
"""
import sys
import os
import argparse
import paramiko
import select
import time

HOST = "124.71.238.180"
PORT = 22
USER = "root"
PASS = "huaweiyun123+-hd"
REMOTE_BASE = "/root/EarnHFT-main/EarnHFT_Algorithm"


def get_client():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, USER, PASS, timeout=15)
    return c


def run_cmd(cmd, timeout=600, print_output=True):
    """Run a shell command, return (stdout_str, stderr_str, exit_code)."""
    client = get_client()
    full_cmd = f"cd {REMOTE_BASE} && {cmd}"
    stdin, stdout, stderr = client.exec_command(full_cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    client.close()
    if print_output:
        if out:
            print(out, end="")
        if err:
            print(err, end="", file=sys.stderr)
    return out, err, code


def download(remote_path, local_path):
    client = get_client()
    sftp = client.open_sftp()
    sftp.get(remote_path, local_path)
    sftp.close()
    client.close()
    print(f"Downloaded {remote_path} -> {local_path}")


def upload(local_path, remote_path):
    client = get_client()
    sftp = client.open_sftp()
    sftp.put(local_path, remote_path)
    sftp.close()
    client.close()
    print(f"Uploaded {local_path} -> {remote_path}")


def run_script(local_script):
    """Upload a local script and execute it on the server."""
    remote_tmp = f"/tmp/_tmp_script_{os.path.basename(local_script)}"
    upload(local_script, remote_tmp)
    run_cmd(
        f"chmod +x {remote_tmp} && bash {remote_tmp} && rm -f {remote_tmp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", help="Shell command to run")
    parser.add_argument("-f", "--script", help="Local script to upload & run")
    parser.add_argument("--get", nargs=2, metavar=("REMOTE",
                        "LOCAL"), help="Download file")
    parser.add_argument("--put", nargs=2, metavar=("LOCAL",
                        "REMOTE"), help="Upload file")
    parser.add_argument("-t", "--timeout", type=int, default=600)
    args = parser.parse_args()

    if args.get:
        download(args.get[0], args.get[1])
    elif args.put:
        upload(args.put[0], args.put[1])
    elif args.script:
        run_script(args.script)
    elif args.command:
        _, _, code = run_cmd(args.command, timeout=args.timeout)
        sys.exit(code)
    else:
        parser.print_help()
