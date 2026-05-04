import subprocess
import sys
import os

def main():
    if len(sys.argv) < 3:
        print("Usage: python run_harvest.py <username> <bearer_token>")
        sys.exit(1)

    username = sys.argv[1]
    bearer_token = sys.argv[2]

    output_dir = "tweets-data"
    os.makedirs(output_dir, exist_ok=True)

    search_keyword = f"from:{username} lang:id"
    output_file = f"{username}.csv"

    command = [
        "npx.cmd",  # untuk Windows
        "tweet-harvest",
        "-t", bearer_token,
        "-s", search_keyword,
        "-o", output_file,
        "--limit", "10"
    ]

    print(f"[INFO] Menjalankan scraping tweet untuk @{username}...")
    print(f"[INFO] Menyimpan sementara ke: {output_file}")

    try:
        result = subprocess.run(command, capture_output=True, text=True)

        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("[ERROR OUTPUT]")
            print(result.stderr)

        final_path = os.path.join(output_dir, f"{username}.csv")
        if os.path.exists(output_file):
            os.replace(output_file, final_path)
            print(f"[INFO] File berhasil dipindahkan ke: {final_path}")
        else:
            print("[ERROR] File hasil scraping tidak ditemukan.")
            sys.exit(1)

        print("[INFO] Tweet berhasil diambil dan disimpan.")

    except FileNotFoundError:
        print("[ERROR] 'npx' tidak ditemukan. Pastikan Node.js sudah terinstal.")
        sys.exit(1)

if __name__ == "__main__":
    main()