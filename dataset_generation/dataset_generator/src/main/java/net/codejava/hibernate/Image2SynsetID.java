package net.codejava.hibernate;

import java.io.Serializable;
import java.util.Objects;

import javax.persistence.Column;
import javax.persistence.Embeddable;

@Embeddable
public class Image2SynsetID implements Serializable {
	private static final long serialVersionUID = 1L;
	@Column(name="synset_id")
	Long synset_id;
	
	@Column(name="image_id")
	Long image_id;
	
	public Image2SynsetID() {
		super();
		// TODO Auto-generated constructor stub
	}
	

	public Image2SynsetID(Long synset_id, Long image_id) {
		super();
		this.synset_id = synset_id;
		this.image_id = image_id;
	}

	
	public Long getSynset_id() {
		return synset_id;
	}


	public void setSynset_id(Long synset_id) {
		this.synset_id = synset_id;
	}


	public Long getImage_id() {
		return image_id;
	}


	public void setImage_id(Long image_id) {
		this.image_id = image_id;
	}
		

	@Override
	public String toString() {
		return "Image2SynsetID [synset_id=" + synset_id + ", image_id=" + image_id + "]";
	}


	public boolean equals(Object o){
		if (this == o) return true;
		if (!(o instanceof Image2SynsetID)) return false;
		Image2SynsetID that = (Image2SynsetID) o;
		return Objects.equals(getSynset_id(), that.getSynset_id()) && 
				Objects.equals(getImage_id(), that.getImage_id());
	}
	
	public int hashCode(){
		return Objects.hash(getSynset_id(), getImage_id());
	}
}
