package net.codejava.hibernate;

import javax.persistence.*;

@Entity
@Table(name="image2Synset")
public class Image2Synset{
	@EmbeddedId
	private Image2SynsetID id;
	private Long source_id;
	private Boolean main;
	
	public Image2Synset() {
		super();
		// TODO Auto-generated constructor stub
	}

	public Image2Synset(Image2SynsetID id, Long source_id, Boolean main) {
		super();
		this.id = id;
		this.source_id = source_id;
		this.main = main;
	}

	public Image2SynsetID getId() {
		return id;
	}

	public void setId(Image2SynsetID id) {
		this.id = id;
	}

	public Long getSource_id() {
		return source_id;
	}

	public void setSource_id(Long source_id) {
		this.source_id = source_id;
	}

	public Boolean isMain() {
		return main;
	}

	public void setMain(Boolean main) {
		this.main = main;
	}

	@Override
	public String toString() {
		return "Image2Synset [id=" + id + ", source_id=" + source_id + ", main=" + main + "]";
	}
	
	
}
