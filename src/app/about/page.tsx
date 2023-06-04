"use client"
 
import React  from 'react'
import Image from "next/image"

async function getImg( ) {
     const img = await fetch("http://localhost:3000/api/about", {cache:"no-store"});
     console.log(img)
     return img
}

export default async function Page() {
 const a = await getImg()
 const base64=await a.text()
  return (
    <div className='w-full h-full flex justify-center items-center'>
      <Image src={base64}  
      alt="img" height={512} width={512} />
    </div>
  )
}
