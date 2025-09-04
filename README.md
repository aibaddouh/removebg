# removebg
Remove backgrounds from one or more images using rembg.

First run will download and install required dependencies in the user's home directory. Subsequent runs will reuse the environment.

Usage examples:
- GUI mode:
```sh
./removebg.py
```

- Remove background from a single file:
```sh
./removebg.py /path/to/image.jpg
```

- Multiple files and URLs, save to custom folder:
```sh
./removebg.py img1.jpg https://example.com/cat.png --outdir ./out
```

- Overwrite existing outputs:
```sh
./removebg.py img1.jpg img2.png --overwrite
```
