extern crate primal;

use primal::Sieve;

fn num_divisors(n: usize, primes: &Sieve) -> Option<usize> {
    match primes.factor(n) {
        Ok(factors) => Some(factors.into_iter().fold(1, |acc, (_, x)| acc * (x + 1))),
        Err(_) => None,
    }
}

fn main() {
    let sieve = Sieve::new(10000);
    let suspect = 5273;
    println!("{} is prime: {}", suspect, sieve.is_prime(suspect));
    let not_is_prime = 1024;
    println!("{} is prime: {}", not_is_prime, sieve.is_prime(not_is_prime));
//    println!("Hello, world!");
    println!("Factors of 2610: \n");
    println!("{:?}", sieve.factor(2610));
    println!("{:?}", num_divisors(2610, &sieve));
}
