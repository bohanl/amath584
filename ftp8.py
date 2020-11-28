

def ftp8_all(m=4,n=4):
    expo = [i for i in range(2**m)]
    logd = [i for i in range(2**n)]

    ftps = []
    for e in expo:
        for d in logd:
            t = [ 2**(-i-1) * (1 if d & (2**b) else 0) for i, b in enumerate(range(n-1, -1, -1))]
            ftp = 2**(-e) * (1+sum(t))
            if ftp > 1:
                print(f'info: e={e}, d={d}')
            ftps.append(ftp)

    return sorted(ftps)


def main():
    ftps = ftp8_all()
    for f in ftps:
        print('{:14.15f}'.format(f))
    print(f'Total = {len(ftps)}')


if __name__ == '__main__':
    main()

