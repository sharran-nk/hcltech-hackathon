# AWS setup: private S3 bucket + least-privilege IAM

Replace `YOUR_BUCKET` (globally unique, lowercase) and `ap-south-1` with your values.
Never put AWS access keys in code, `.env`, chat, or Git. On EC2 the IAM **role** supplies
temporary credentials automatically.

## 1. Create the private bucket
```bash
aws s3api create-bucket --bucket YOUR_BUCKET --region ap-south-1 \
  --create-bucket-configuration LocationConstraint=ap-south-1

# Block ALL public access (the important one)
aws s3api put-public-access-block --bucket YOUR_BUCKET \
  --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

# Default encryption at rest (the app also requests AES256 on every upload)
aws s3api put-bucket-encryption --bucket YOUR_BUCKET \
  --server-side-encryption-configuration '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
```
(For us-east-1 omit `--create-bucket-configuration`.)

Verify:
```bash
aws s3api get-public-access-block --bucket YOUR_BUCKET   # all four values must be true
```

## 2. Least-privilege IAM policy
Only Put/Get/DeleteObject, only on the `documents/` prefix of this one bucket. No list, no
bucket-level actions, no wildcard on other buckets. Save as `solstice-s3-policy.json`:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DocumentObjectsOnly",
      "Effect": "Allow",
      "Action": ["s3:PutObject", "s3:GetObject", "s3:DeleteObject"],
      "Resource": "arn:aws:s3:::YOUR_BUCKET/documents/*"
    }
  ]
}
```
Because `s3:ListBucket` is not granted, S3 returns 403 (not 404) for missing keys. The app
accounts for this and never needs to list the bucket.

## 3. Role for the EC2 instance
```bash
# trust policy: lets EC2 assume the role
cat > ec2-trust.json <<'EOF'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}
EOF

aws iam create-role --role-name solstice-ec2-role --assume-role-policy-document file://ec2-trust.json
aws iam put-role-policy --role-name solstice-ec2-role --policy-name solstice-s3-access \
  --policy-document file://solstice-s3-policy.json
aws iam create-instance-profile --instance-profile-name solstice-ec2-profile
aws iam add-role-to-instance-profile --instance-profile-name solstice-ec2-profile --role-name solstice-ec2-role
```
Then attach `solstice-ec2-profile` to the instance (Console: EC2 > instance > Actions >
Security > Modify IAM role). Phase 11 walks through this again step by step.

## 4. Verify from the instance
```bash
aws sts get-caller-identity          # should show the assumed role, not a user
python scripts/s3_smoke_test.py      # upload / download / delete one tiny object
```
Expect `upload OK`, `download OK, content matches: True`, `delete OK`.

## 5. Notes for production (future work)
- Bucket versioning + lifecycle rules, S3 access logging / CloudTrail data events.
- A VPC gateway endpoint for S3 so traffic never leaves AWS's network.
- SSE-KMS with a customer-managed key instead of AES256.
