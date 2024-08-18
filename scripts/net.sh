#!/bin/sh

wget_or_curl=$((command -v wget &>/dev/null && echo "wget -q") || \
               (command -v curl &>/dev/null && echo "curl -L -s -k"))

if [ -z "$wget_or_curl" ]; then
  >&2 printf '%s\n' "Neither wget or curl is installed." \
	         "Install one of these tools to download NNUE files automatically."
  exit 1
fi

sha256sum=$((command -v shasum &>/dev/null && echo "shasum -a 256") || \
            (command -v sha256sum &>/dev/null && echo "sha256sum"))

if [ -z "$sha256sum" ]; then
  >&2 echo "sha256sum not found, NNUE files will be assumed valid."
fi

function get_nnue_filename {
  grep $1 evaluate.h | grep "#define" | sed 's/.*\(nn-[a-z0-9]\{12\}.nnue\).*/\1/'
}

function validate_network {
  # If no sha256sum command is available, assume the file is always valid.
  if [ ! -z "$sha256sum" -a -f "$1" ]; then
    if [ "$1" != "nn-$($sha256sum $1 | cut -c 1-12).nnue" ]; then
      rm -f "$1"
      return 1
    fi
  fi
}

function fetch_network {
  local filename=$(get_nnue_filename $1)

  if [ -z "$filename" ]; then
    >&2 echo "NNUE file name not found for: $1"
    return 1
  fi

  if [ -f "$filename" ]; then
    if validate_network $filename; then
      echo "Existing $filename validated, skipping download"
      return
    else
      echo "Removing invalid NNUE file: $filename"
    fi
  fi

  local download_url=(
    "https://tests.stockfishchess.org/api/nn/$filename"
    "https://github.com/official-stockfish/networks/raw/master/$filename"
  )

  for url in "${download_url[@]}"; do
    echo "Downloading from $url ..."
    if $wget_or_curl $url; then
      if validate_network $filename; then
        echo "Successfully validated $filename"
      else
        echo "Downloaded $filename is invalid"
        continue
      fi
    else
      echo "Failed to download from $url"
    fi
    if [ -f $filename ]; then
      return
    fi
  done

  # Download was not successful in the loop, return false.
  >&2 echo "Failed to download $filename"
  return 1
}

fetch_network EvalFileDefaultNameBig && \
fetch_network EvalFileDefaultNameSmall
